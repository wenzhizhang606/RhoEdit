"""Generate in a disposable process; judge saved answers in the CPU-only parent."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import uuid


SEED = 69420


def get_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--edited_model_dir', required=True)
    parser.add_argument('--data_type', required=True,
                        choices=['zsre', 'counterfact', 'multi_counterfact', 'wiki',
                                 'safeedit_train', 'safeedit_test', 'zsre10k'])
    parser.add_argument('--eval_num', type=int, default=3000)
    parser.add_argument('--max_length', type=int, default=40)
    parser.add_argument('--context_type', required=True, choices=['qa_inst', 'chat_temp', 'no_context'])
    parser.add_argument('--alg_name', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--evaluation_criteria', required=True, choices=['exact_match', 'llm_judge'])
    parser.add_argument('--wandb_project', default='CrispEdit_EVAL')
    parser.add_argument('--wandb_run_id', default=None)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--judge_batch_size', type=int, default=16)
    parser.add_argument('--judge_only', action='store_true',
                        help='Reuse a completed generation from this script, without loading a model.')
    parser.add_argument('--_generation_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_generation_run_id', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.eval_num <= 0 or args.max_length <= 0 or args.judge_batch_size <= 0:
        parser.error('eval_num, max_length and judge_batch_size must be positive')
    if args._generation_worker and (not args._generation_run_id or args.judge_only):
        parser.error('Invalid generation worker arguments')
    return args


def build_hparams_from_args(args):
    return SimpleNamespace(
        alg_name=args.alg_name, context_type=args.context_type, max_length=args.max_length,
        api_key=os.getenv('API_KEY'), base_url=os.getenv('BASE_URL'), model=os.getenv('MODEL'),
        evaluation_type='WILD', model_name=args.model_name,
        evaluation_criteria=args.evaluation_criteria, data_type=args.data_type,
    )


def evaluation_name(args):
    return args.edited_model_dir + f'_eval_{args.evaluation_criteria}_{args.context_type}'


def generation_config(args):
    return {name: getattr(args, name) for name in (
        'edited_model_dir', 'model_name', 'alg_name', 'data_type', 'context_type',
        'evaluation_criteria', 'eval_num', 'max_length',
    )}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=4)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def evaluation_lock(log_dir):
    """Avoid concurrent writers to the same result directory on Linux."""
    with (log_dir / '.evaluation.lock').open('a') as lock:
        if os.name == 'posix':
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Another evaluation owns {log_dir}') from exc
        yield


def _group_alive(group_id):
    # Reparented zombies no longer own CUDA contexts or memory.
    if sys.platform.startswith('linux'):
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == group_id and fields[0] not in ('Z', 'X'):
                    return True
            except (FileNotFoundError, ProcessLookupError):
                continue
        return False
    try:
        os.killpg(group_id, 0)
        return True
    except ProcessLookupError:
        return False


def stop_generation_group(process, grace_seconds=5.0):
    """Stop only this child's process group, including its vLLM workers."""
    if os.name != 'posix':
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=grace_seconds)
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            process.poll()
            if not _group_alive(process.pid):
                break
            time.sleep(.1)
        if not _group_alive(process.pid):
            break
    else:
        raise RuntimeError('Generation workers have not exited; refusing to start the judge')
    process.wait(timeout=grace_seconds)


def run_generation_process(command, env):
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    # A separate session protects workers from terminal signals. Forward parent
    # termination through the same cleanup path so those workers are not orphaned.
    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    process = None
    try:
        process = subprocess.Popen(command, env=env, start_new_session=(os.name == 'posix'))
        returncode = process.wait()
    finally:
        try:
            if process is not None:
                stop_generation_group(process)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def get_model_and_tokenizer_from_dir(edited_model_dir_local):
    from transformers import AutoTokenizer
    from vllm import LLM

    prefix = os.getenv('HF_CACHE_DIR')
    if not prefix:
        raise RuntimeError('HF_CACHE_DIR is not set')
    edited_model_dir = prefix + edited_model_dir_local
    print(f'[vLLM] Loading model from: {edited_model_dir}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(edited_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = LLM(model=edited_model_dir, tokenizer=edited_model_dir, dtype='bfloat16',
                trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    return model, tokenizer


def run_generation(args, hparams, log_dir):
    import random
    import numpy as np
    import torch
    from tqdm import tqdm
    from utils import prepare_requests_from_data_type
    from easyeditor.evaluate.evaluate import compute_edit_quality, compute_edit_quality_safety

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    state = {'run_id': args._generation_run_id, 'state': 'running',
             'config': generation_config(args), 'cases': 0}
    state_path = log_dir / 'generation_state.json'
    result_path = log_dir / 'results_pending_judge.json'
    atomic_json(state_path, state)
    requests = prepare_requests_from_data_type(args.data_type)
    if not requests:
        raise ValueError('No evaluation requests')
    model, tokenizer = get_model_and_tokenizer_from_dir(args.edited_model_dir)
    # Preserve the previous ordering: model initialization precedes sampling.
    requests = random.sample(requests, len(requests))[:args.eval_num]
    evaluate = compute_edit_quality_safety if 'safe' in args.data_type else compute_edit_quality
    all_metrics = []
    for i, request in enumerate(tqdm(requests)):
        all_metrics.append({
            'case_id': i, 'requested_rewrite': request, 'pre': {},
            'post': evaluate(model, hparams.model_name, hparams, tokenizer, request, 0),
        })
        atomic_json(result_path, all_metrics)
    state.update(state='complete', cases=len(all_metrics),
                 sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
    atomic_json(state_path, state)
    print(f'[generation] Saved {len(all_metrics)} cases; exiting GPU process.', flush=True)
    # Keep model references alive until os._exit; old vLLM destructors/atexit
    # handlers can otherwise hang before the parent gets control back.
    return model, tokenizer


def load_completed_generation(args, log_dir, run_id=None):
    state_path = log_dir / 'generation_state.json'
    if not state_path.exists():
        raise RuntimeError('Missing generation completion record; cannot verify saved answers are complete')
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if state.get('state') != 'complete' or state.get('config') != generation_config(args):
        raise RuntimeError('Generation incomplete or evaluation arguments differ from saved generation')
    if run_id is not None and state.get('run_id') != run_id:
        raise RuntimeError('Completion record belongs to an older generation')
    raw = (log_dir / 'results_pending_judge.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != state.get('sha256'):
        raise RuntimeError('Saved generation hash mismatch')
    data = json.loads(raw)
    if not isinstance(data, list) or not data or len(data) != state.get('cases'):
        raise RuntimeError('Saved generation count mismatch')
    return data


def run_judging(args, hparams, log_dir, all_metrics):
    # Original parent; GPUs are hidden before any EasyEdit or torch import.
    from easyeditor.editors.utils import summary_metrics
    from easyeditor.tools import ExperimentTracker

    config = {key: value for key, value in vars(hparams).items() if key not in ('api_key', 'base_url')}
    ExperimentTracker.init(project=args.wandb_project, name=evaluation_name(args),
                           config=config, tracker_type='wandb', mode=not args.no_wandb)
    try:
        if args.evaluation_criteria == 'llm_judge':
            from easyeditor.evaluate.evaluate_utils import resolve_pending_llm_judges
            print(f'[llm_judge] Starting in CPU-only parent pid={os.getpid()}.', flush=True)
            all_metrics = resolve_pending_llm_judges(
                all_metrics, hparams.api_key, batch_size=args.judge_batch_size,
            )
        summary_metrics(all_metrics, str(log_dir))
        print(f'[evaluation] Results saved to {log_dir}', flush=True)
    finally:
        ExperimentTracker.finish()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = get_arguments(argv)
    from dotenv import load_dotenv
    load_dotenv()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    hparams = build_hparams_from_args(args)
    log_dir = Path('./logs') / evaluation_name(args)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args._generation_worker:
        try:
            resources = run_generation(args, hparams, log_dir)
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # Persisted JSON is the handoff; no model crosses it.

    generation_env = os.environ.copy()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with evaluation_lock(log_dir):
        run_id = None
        if not args.judge_only:
            run_id = uuid.uuid4().hex
            command = [sys.executable, '-u', str(Path(__file__).resolve()), *argv,
                       '--_generation_worker', '--_generation_run_id', run_id]
            print('[generation] Starting a dedicated GPU process.', flush=True)
            run_generation_process(command, generation_env)
            print('[generation] GPU process and its workers exited; loading saved answers.', flush=True)
        all_metrics = load_completed_generation(args, log_dir, run_id)
        run_judging(args, hparams, log_dir, all_metrics)


if __name__ == '__main__':
    main()
