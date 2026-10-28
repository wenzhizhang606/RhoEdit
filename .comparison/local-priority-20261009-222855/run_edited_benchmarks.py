import argparse
import gc
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 69420
random.seed(SEED)
np.random.seed(SEED)


def get_model_and_tokenizer_from_dir(edited_model_dir_local):
    # Import vLLM only in the generation worker. The parent stays CPU-only.
    from transformers import AutoTokenizer
    from vllm import LLM

    prefix_dir = os.getenv("HF_CACHE_DIR")
    if not prefix_dir:
        raise RuntimeError("HF_CACHE_DIR is not set")
    edited_model_dir = prefix_dir + edited_model_dir_local
    print(f"[vLLM] Loading model from: {edited_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        edited_model_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = LLM(
        model=edited_model_dir,
        tokenizer=edited_model_dir,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
    return model, tokenizer


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edited_model_dir", required=True, type=str, default=None,
                        help="Path to edited model for evaluation.")
    parser.add_argument("--data_type", required=True, type=str, default="zsre",
                        choices=["zsre", "counterfact", "multi_counterfact", "wiki",
                                 "safeedit_train", "safeedit_test", "zsre10k"])
    parser.add_argument("--eval_num", required=False, type=int, default=3000,
                        help="Number of evaluation instances to use.")
    parser.add_argument("--max_length", required=False, type=int, default=40,
                        help="Maximum length of generated sequences.")
    parser.add_argument("--context_type", required=True, type=str, default="qa_inst",
                        choices=["qa_inst", "chat_temp", "no_context"],
                        help="Type of context to use for evaluation.")
    parser.add_argument("--alg_name", required=True, type=str, default="ft_edit",
                        help="Name of the editing algorithm used.")
    parser.add_argument("--model_name", required=True, type=str, default="gpt2-xl",
                        help="Name of the base model used.")
    parser.add_argument("--evaluation_criteria", required=True, type=str,
                        default="exact_match", choices=["exact_match", "llm_judge"],
                        help="Evaluation criteria to use.")
    parser.add_argument("--wandb_project", type=str, default="CrispEdit_EVAL",
                        help="WandB project name.")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="WandB run ID for resuming runs.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging.")
    parser.add_argument("--judge_batch_size", required=False, type=int, default=16,
                        help="Number of generated answers to judge per batch.")
    parser.add_argument("--_generation_worker", action="store_true",
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def build_hparams_from_args(args):
    hparams = SimpleNamespace()
    hparams.alg_name = args.alg_name
    hparams.context_type = args.context_type
    hparams.max_length = args.max_length
    hparams.api_key = API_KEY
    hparams.base_url = BASE_URL
    hparams.model = MODEL
    hparams.evaluation_type = "WILD"
    hparams.model_name = args.model_name
    hparams.evaluation_criteria = args.evaluation_criteria
    hparams.data_type = args.data_type
    return hparams


def shutdown_vllm_best_effort(model):
    """Ask vLLM to stop workers before the generation process exits."""
    import torch

    engine = getattr(model, "llm_engine", None)
    executor = getattr(engine, "model_executor", None) if engine is not None else None
    if executor is not None:
        shutdown = getattr(executor, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                print(f"[vLLM] executor.shutdown skipped: {exc}")

    # This is a private/version-dependent API, so process exit remains the
    # reliable cleanup boundary when it is unavailable or partially effective.
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception as exc:
        print(f"[vLLM] destroy_model_parallel skipped: {exc}")

    if engine is not None:
        engine.model_executor = None
        model.llm_engine = None
    del executor, engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_generation_worker(args, hparams, log_dir):
    """Generate all answers in a process whose exit destroys CUDA state."""
    # CUDA is initialized only in this process, never in the CPU judging parent.
    import torch
    from utils import prepare_requests_from_data_type, print_time

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    from easyeditor.evaluate.evaluate import (
        compute_edit_quality,
        compute_edit_quality_safety,
    )

    requests = prepare_requests_from_data_type(args.data_type)
    requests = random.sample(requests, len(requests))
    if args.eval_num is not None:
        requests = requests[:args.eval_num]

    model, tokenizer = get_model_and_tokenizer_from_dir(args.edited_model_dir)
    os.makedirs(log_dir, exist_ok=True)
    result_path = os.path.join(log_dir, "results_pending_judge.json")
    all_metrics = []
    edit_eval_method = (
        compute_edit_quality_safety if "safe" in args.data_type else compute_edit_quality
    )

    print_time("Begin Post Edit Eval Time")
    print("=" * 50)
    print("safe" not in args.data_type)
    try:
        for i, request in enumerate(tqdm(requests)):
            metrics = {
                "case_id": i,
                "requested_rewrite": request,
                "pre": {},
                "post": edit_eval_method(
                    model,
                    hparams.model_name,
                    hparams,
                    tokenizer,
                    request,
                    device=0,
                ),
            }
            all_metrics.append(metrics)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=4)
    finally:
        shutdown_vllm_best_effort(model)
        del tokenizer
        del model
        # Process exit is the hard boundary that destroys vLLM's CUDA state.

    print(f"[vLLM] generated {len(all_metrics)} cases; worker is exiting")


def run_generation_in_subprocess(args):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        *sys.argv[1:],
        "--_generation_worker",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    args = get_arguments()
    hparams = build_hparams_from_args(args)
    run_name = args.edited_model_dir + f"_eval_{args.evaluation_criteria}_{args.context_type}"
    log_dir = f"./logs/{run_name}"

    if args._generation_worker:
        run_generation_worker(args, hparams, log_dir)
        raise SystemExit(0)

    # The parent imports no vLLM model and does no CUDA seeding. It only judges
    # the JSON after the generation subprocess has exited.
    from easyeditor.tools import ExperimentTracker

    ExperimentTracker.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(hparams),
        tracker_type="wandb",
        mode=not args.no_wandb,
    )

    os.makedirs(log_dir, exist_ok=True)
    run_generation_in_subprocess(args)

    result_path = os.path.join(log_dir, "results_pending_judge.json")
    with open(result_path, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)

    from easyeditor.editors.utils import summary_metrics
    from utils import print_time

    if args.evaluation_criteria == "llm_judge":
        from easyeditor.evaluate.evaluate_utils import resolve_pending_llm_judges

        all_metrics = resolve_pending_llm_judges(
            all_metrics,
            hparams.api_key,
            batch_size=args.judge_batch_size,
        )

    summary_metrics(all_metrics, log_dir)
    print_time("End Post Edit Eval Time")
    ExperimentTracker.finish()
