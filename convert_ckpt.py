#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from glob import glob
from collections import defaultdict
import os
import shutil
import torch.distributed.tensor

parser = argparse.ArgumentParser(description="")
parser.add_argument("--root", type=str)
parser.add_argument("--step", type=int, default=80)
parser.add_argument("--world_size", type=int, default=8)
args = parser.parse_args()

def main():
    fsdp_checkpoint_path = os.path.join(args.root, f"global_step_{args.step}/actor")
    huggingface_model_path = os.path.join(args.root, f"global_step_{args.step}/actor/huggingface")
    output_path = os.path.join(args.root, f"huggingface_checkpoint/checkpoint_global_step_{args.step}")
    state_dict = defaultdict(list)

    for rank in range(args.world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{args.world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath, weights_only=False)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)

    score_src_file = os.path.join(fsdp_checkpoint_path, 'score_module.pt')
    score_dest_file = os.path.join(output_path, 'score_module.pt')
    shutil.copy(score_src_file, score_dest_file)


if __name__ == "__main__":
    main()