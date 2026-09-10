import json
import argparse
import torch
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_DIR")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import wandb
from utils import prepare_prompts_from_data_type, save_model_and_tokenizer
import random

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def resolve_local_model_name(model_name):
    """Match server run_crispedit.py: load from HF_CACHE_DIR + hub id when that folder exists."""
    if not model_name:
        return model_name
    cache = os.getenv("HF_CACHE_DIR") or ""
    candidates = [model_name]
    if cache:
        candidates.append(cache + model_name)
        candidates.append(os.path.join(cache.rstrip("/"), model_name))
        candidates.append(os.path.join(cache.rstrip("/"), os.path.basename(str(model_name).rstrip("/"))))
        if "/" not in str(model_name).strip("/"):
            candidates.append(cache + "meta-llama/" + model_name)
    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isfile(os.path.join(path, "config.json")):
            print(f"Using local model: {path}")
            return path
    return model_name


from easyeditor import (
    FTHyperParams,
    MENDHyperParams,
    UltraEditHyperParams,
    ROMEHyperParams,
    R_ROMEHyperParams,
    MEMITHyperParams,
    GraceHyperParams,
    WISEHyperParams,
    AlphaEditHyperParams,
    IKEHyperParams,
    MELOHyperParams,
    LoRAHyperParams,
    BaseEditor,
)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str, default='zsre', choices=['zsre', 'counterfact', 'wiki', 'zsre10k'])
    parser.add_argument('--editing_method', required=True, type=str, choices=['FT', 'MEND', 'ROME', 'R-ROME', 'MEMIT', 'GRACE', 'WISE', 'AlphaEdit', 'IKE', 'MELO', 'LoRA', 'UltraEdit'])
    parser.add_argument('--eval_every', required=True, type=int, default=512, help='Evaluation frequency.')
    parser.add_argument('--sequential_edit', default='True', type=str)
    parser.add_argument('--batch_edit', default='False', type=str)
    parser.add_argument('--num_edits', type=int, default=100, help='Sequential edit batch. Only used if sequential_edit is True.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for fine-tuning in a sequential chunk. CAUTION: THIS IS HARDLY USED. MAKE SURE YOU KNOW WHAT YOU ARE DOING.')
    parser.add_argument('--wandb_project', type=str, default='CrispEdit', help='WandB project name.')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging.')
    parser.add_argument(
        '--skip_task1_loss',
        action='store_true',
        help='Skip Wikipedia Task-1 loss. Implied by --no_wandb unless --task1_loss is set.',
    )
    parser.add_argument(
        '--task1_loss',
        action='store_true',
        help='Force Wikipedia Task-1 loss even when --no_wandb is set.',
    )
    args = parser.parse_args()
    return args


def configure_device_placement(hparams, editing_method):
    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 0:
        raise RuntimeError('No CUDA device is visible. These editing methods require CUDA.')

    use_model_parallel = visible_gpus >= 2
    primary_device = 0

    hparams.model_parallel = use_model_parallel
    hparams.device = primary_device
    if use_model_parallel:
        # Keep GPU 0 relatively free because editing methods place auxiliary
        # matrices and optimizer state on the primary device.
        hparams.device_map = 'balanced_low_0'

    # MEND's hypernetwork adds a sizeable fp32 allocation. Load the base model
    # in half precision from the start in both single- and multi-GPU modes.
    if editing_method == 'MEND':
        hparams.fp16 = True

    mode = 'multi-GPU' if use_model_parallel else 'single-GPU'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')
    print(
        f'Using {mode}: visible={visible}, count={visible_gpus}, '
        f'primary=cuda:{primary_device}'
    )


def get_hparams_and_editor(args):
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'UltraEdit':
        editing_hparams = UltraEditHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'R-ROME':
        editing_hparams = R_ROMEHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'MELO':
        editing_hparams = MELOHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    else:
        raise NotImplementedError
    
    hparams = editing_hparams.from_hparams(f"./hparams/{args.editing_method}/{args.model}")
    hparams.model_name = resolve_local_model_name(hparams.model_name)
    configure_device_placement(hparams, args.editing_method)
    hparams.batch_size = args.num_edits ### NOTE: We try to match the naming convention in easy edit. batch_size here means the number of edits in a sequential edit.
    hparams.chunk_batch_size = args.batch_size ### NOTE: chunk_batch_size is the actual batch size for fine-tuning in a sequential chunk. Most methods in easyeditor do not use this parameter, so changing this will hardly affect anything.
    assert hparams.chunk_batch_size == 1 or (hparams.chunk_batch_size > 1 and args.editing_method in ['LoRA']), "Currently only LoRA supports batch fine-tuning. Are you sure what you are doing?"
    editor = BaseEditor.from_hparams(hparams)
    return hparams, editor

if __name__ == "__main__":
    args = get_arguments()
    prompts, rephrase_prompts, subject, target_new, locality_inputs, ground_truth = prepare_prompts_from_data_type(args.data_type)
    hparams, editor = get_hparams_and_editor(args)
    save_model_name = f"{args.model}_{args.editing_method}_{args.data_type}"
    print(f"Model will be saved to BASE_DIR/{save_model_name}")
    wandb.init(project=args.wandb_project, name=save_model_name, config=vars(hparams), mode="online" if not args.no_wandb else "disabled")

    if args.sequential_edit == "True" or args.sequential_edit == "true":
        sequential_edit = True
    else:
        sequential_edit = False

    if args.batch_edit == "True" or args.batch_edit == "true":
        batch_edit = True
    else:
        batch_edit = False

    skip_task1_loss = args.skip_task1_loss or (args.no_wandb and not args.task1_loss)
    if skip_task1_loss:
        print("Skipping Wikipedia Task-1 loss. Pass --task1_loss to keep it.")

    if batch_edit:
        edited_model, tokenizer = editor.batch_edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            subject=subject,
            target_new=target_new,
            locality_inputs=locality_inputs,
            eval_every=args.eval_every,
            skip_task1_loss=skip_task1_loss,
        )
    else:
        edited_model, tokenizer = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            subject=subject,
            target_new=target_new,
            locality_inputs=locality_inputs,
            sequential_edit=sequential_edit,
            eval_every=args.eval_every,
            skip_task1_loss=skip_task1_loss,
        )

    save_model_and_tokenizer(edited_model, tokenizer, save_model_name)
