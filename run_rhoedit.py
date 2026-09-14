import random
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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

    hparams.edit_n_samples = args.edit_sample_num
    hparams.recalculate_cache = args.recalculate_cache
    hparams.recalculate_weight_threshold = args.recalculate_weight_threshold
    hparams.edit_cache_style = args.edit_cache_style

    hparams.disable_old_loss_check = args.disable_old_loss_check


    if args.sequential_edit:
        assert args.num_edits >= args.batch_size, \
            "Makes no sense to have a batch_size bigger than number of edits..."
        hparams.num_edits = args.num_edits

    return hparams


def calculate_model_name(args, hparams):
    name = (f"{args.model}_rhoedit_adam_{args.data_type}"
                        f"_{hparams.task_mom2_dataset}"
                        f"_{hparams.newton_damping}_{hparams.soft_lambda}_{hparams.lr}_12_19")

    if args.sequential_edit:
        name += f"_sequential_{args.num_edits}"
    
    if hparams.recalculate_cache:
        name += f"_recalc_cache_{args.recalculate_weight_threshold}_edit_sample_{args.edit_sample_num}"
    if args.sequential_edit:
        name += f"_edit_cache_{args.edit_cache_style}"

    return name.replace('.', '_')

if __name__ == "__main__":
    args = get_arguments()
    requests = prepare_requests_from_data_type(args.data_type)
    requests = setup_requests_for_safeedit(requests)
    hparams = get_hparams(args)

    save_model_name = calculate_model_name(args, hparams)
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

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    device_map = {
        "model.embed_tokens": 1,
        "model.rotary_emb": 1,
    }

    for layer in range(config.num_hidden_layers):
        device_map[f"model.layers.{layer}"] = 0 if  12 <= layer <= 19 else 1

    device_map["model.norm"] = 1
    device_map["lm_head"] = 1

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device_map,
        local_files_only=True,
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
