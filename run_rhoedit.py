import random
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
# Contiguous allocator helps large K-FAC work sit next to the 7B shards.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from easyeditor.models.rhoedit.utils import (
    KFAC_DATASET_ALIASES,
    normalize_kfac_dataset,
)
from utils import (
    print_time, 
    prepare_requests_from_data_type, 
    save_model_and_tokenizer, 
)

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_DIR")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.models.rhoedit import (
    AdamHyperParams,
    execute_sft_adam,
    execute_sft_adam_sequential,
    setup_requests_for_safeedit,
    update_model_and_tokenizer_with_appropriate_padding_token,
)

from easyeditor.tools import ExperimentTracker


SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, type=str,
                        help='Model name or path, e.g. meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--data_type', required=True, type=str, default='zsre',
                        choices=['zsre', 'zsre10k', 'counterfact', 'wiki',
                                 'safeedit_train', 'safeedit_test'])

    parser.add_argument('--cache_sample_num', type=int, default=10000,
                        help='Number of samples to use for caching projection matrices.')
    parser.add_argument('--cache_task_sample_num', type=int, default=3000,
                        help='Number of samples used for caching editing projection matrices.')
    parser.add_argument('--task_mom2_dataset', type=str, default=None,
                        help='Edit-curvature K-FAC corpus. Default: align with --data_type. '
                             'Pass yaml to keep the YAML value, or a dataset name to override.')
    # Whether to monitor the loss on the original task.                  
    parser.add_argument('--disable_old_loss_check', action='store_true',
                        help='Disable old loss check to speed up sequential editing.')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for fine-tuning.')

    # Sequential
    # 每个阶段编辑的数量
    parser.add_argument('--sequential_edit', action='store_true',
                        help='Whether to use sequential editing. Default is False.')
    parser.add_argument('--num_edits', type=int, default=100,
                        help='Sequential edit batch size.')
    # 模型编辑一段时间后，最初计算的 K-FAC 投影矩阵可能不再适合当前模型
    # 相对参数变化超过约 25% 时触发
    parser.add_argument('--edit_sample_num', type=int, default=1000,
                        help='Request K-FAC sample size (hparams.edit_n_samples) during sequential editing.')
    parser.add_argument('--recalculate_cache', action='store_true',
                        help='Whether to recalculate the projection caches. Default is False.')
    parser.add_argument('--recalculate_weight_threshold', type=float, default=0.25,
                        help='Threshold for recalculating weight projection caches. [0.0-1.0]')
    # 顺序编辑时如何保护历史编辑
    parser.add_argument('--edit_cache_style', type=str, default='mix',
                        choices=['sequential', 'mix', 'disable'],
                        help='Cache style during sequential editing.')
    
    # wandb/swanlab
    parser.add_argument('--wandb_project', type=str, default='RhoEdit',
                        help='project name.')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging.')
    parser.add_argument('--plat_name', type=str, default='swanlab',
                       choices=['swanlab','wandb','none'])


    parser.add_argument('--newton_damping', type=float, default=None,
                        help='Override YAML newton_damping when set.')
    parser.add_argument('--soft_lambda', type=float, default=None,
                        help='Override YAML soft_lambda when set.')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override YAML lr when set.')
    parser.add_argument(
        '--layers',
        type=str,
        default=None,
        help='Comma-separated edit layer indices, e.g. 14,15,16,17,18,19,20,21,22,23.',
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='After the sequential train shuffle (seed 69), keep only the first N '
             'requests. N=100 is one Algorithm-1 chunk, for hyperparameter screening.',
    )

    args = parser.parse_args()
    return args

def get_hparams(args):
    print("[rhoedit_adam] Load RhoEdit Adam configuration.")
    hparams = AdamHyperParams.from_hparams(f"./hparams/RhoEdit/{args.model}")

    hparams.batch_size = args.batch_size
    hparams.mom2_n_samples = args.cache_sample_num
    hparams.task_mom2_n_samples = args.cache_task_sample_num
    yaml_task_dataset = hparams.task_mom2_dataset
    if args.task_mom2_dataset == "yaml":
        chosen_task = yaml_task_dataset
    elif args.task_mom2_dataset:
        chosen_task = args.task_mom2_dataset
    elif args.data_type.lower() in KFAC_DATASET_ALIASES:
        chosen_task = args.data_type
    else:
        chosen_task = yaml_task_dataset
    hparams.task_mom2_dataset = normalize_kfac_dataset(chosen_task)
    print(
        f"[RhoEdit] data_type={args.data_type}  "
        f"K-FAC task dataset={hparams.task_mom2_dataset}"
        f"{'' if hparams.task_mom2_dataset == yaml_task_dataset else f' (YAML was {yaml_task_dataset})'}"
    )
    if args.lr is not None:
        hparams.lr = args.lr
    if args.newton_damping is not None:
        hparams.newton_damping = args.newton_damping
    if args.soft_lambda is not None:
        hparams.soft_lambda = args.soft_lambda
    if args.layers:
        hparams.layers = [int(x) for x in args.layers.split(",") if x.strip()]
        if not hparams.layers:
            raise ValueError("--layers parsed to an empty list")
    print(f"[RhoEdit] edit layers={hparams.layers}")

    hparams.edit_n_samples = args.edit_sample_num
    hparams.recalculate_cache = args.recalculate_cache
    hparams.recalculate_weight_threshold = args.recalculate_weight_threshold
    hparams.edit_cache_style = args.edit_cache_style

    hparams.disable_old_loss_check = args.disable_old_loss_check
    hparams.data_seed = SEED
    hparams.max_samples = args.max_samples


    if args.sequential_edit:
        assert args.num_edits >= args.batch_size, \
            "Makes no sense to have a batch_size bigger than number of edits..."
        hparams.num_edits = args.num_edits

    return hparams


def calculate_model_name(args, hparams):
    layer_tag = f"{min(hparams.layers)}_{max(hparams.layers)}"
    name = (f"{args.model}_rhoedit_adam_{args.data_type}"
                        f"_{hparams.task_mom2_dataset}"
                        f"_{hparams.newton_damping}_{hparams.soft_lambda}_{hparams.lr}"
                        f"_n{int(hparams.mom2_n_samples)}_{layer_tag}")

    if args.sequential_edit:
        name += f"_sequential_{args.num_edits}"
    
    if hparams.recalculate_cache:
        name += f"_recalc_cache_{args.recalculate_weight_threshold}_edit_sample_{args.edit_sample_num}"
    if args.sequential_edit:
        name += f"_edit_cache_{args.edit_cache_style}"
    if args.max_samples:
        name += f"_firstchunk_{args.max_samples}"

    return name.replace('.', '_')

if __name__ == "__main__":
    args = get_arguments()
    requests = prepare_requests_from_data_type(args.data_type)
    requests = setup_requests_for_safeedit(requests)
    hparams = get_hparams(args)

    save_model_name = calculate_model_name(args, hparams)
    hparams.run_name = save_model_name
    print(f"Model will be saved to BASE_DIR/{save_model_name}")

    ExperimentTracker.init(project=args.wandb_project, name=save_model_name, config=vars(hparams),
                            tracker_type=args.plat_name,mode = not args.no_wandb)

    MODEL_NAME = hparams.model_name
    if os.path.exists(HF_CACHE_DIR+MODEL_NAME):
        MODEL_NAME=HF_CACHE_DIR+MODEL_NAME
    print(f" Load model path as:{MODEL_NAME}")
    '''
    #最终需要保留
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto',  
                                    local_files_only=True)
    '''

    #qwen2.5
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    # device_map="auto" (contiguous split). Qwen wiki needs ≥2×48GB; Llama
    # sequential is launched with 3. Cross-GPU tensors are staged through
    # host by patch_device_map_host_transfers() because P2P copies on this
    # machine return garbage (all-NaN logits from layer 0).
    n_visible = torch.cuda.device_count()
    if n_visible < 2:
        raise RuntimeError(
            f"RhoEdit device_map=auto needs ≥2 visible GPUs; "
            f"got {n_visible}. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        local_files_only=True,
    )
    dmap = getattr(model, "hf_device_map", {}) or {}
    print(
        f"[RhoEdit] auto device_map visible={n_visible} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"attn={model.config._attn_implementation}"
    )
    for key in (
        "model.embed_tokens",
        "model.layers.0",
        "model.layers.12",
        "model.layers.19",
        "model.layers.27",
        "model.norm",
        "lm_head",
    ):
        if key in dmap:
            print(f"  {key} -> cuda:{dmap[key]}")
    offloaded = sorted({str(v) for v in dmap.values() if v in ("cpu", "disk")})
    if offloaded:
        # Happens when the assigned GPUs are still busy (e.g. overlapping jobs):
        # accelerate silently offloads and the run becomes slow/wrong.
        raise RuntimeError(
            f"device_map=auto offloaded modules to {offloaded}; the assigned GPUs "
            f"are not free. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
    '''
    # llama3
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE_DIR,
        device_map="auto",
        local_files_only=True,
    )
    '''
    # set appropriate padding token
    model, tokenizer = update_model_and_tokenizer_with_appropriate_padding_token(model, tokenizer, hparams)
    
    
    print_time("Begin FT Time")
    if args.sequential_edit:
        edited_model = execute_sft_adam_sequential(model, tokenizer, requests, hparams)
    else:
         edited_model = execute_sft_adam(model, tokenizer, requests, hparams)

        
    print_time("End FT Time")
    save_model_and_tokenizer(edited_model, tokenizer, save_model_name)
