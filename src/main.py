import argparse
import logging
import os
from datetime import datetime
from models import load_model
import json
from prompts import *
from tqdm import tqdm
from gen_meta_prompt import SelfDiscover


def setup_logging(log_file, log_level):
    logger = logging.getLogger("__name__")
    logger.setLevel(getattr(logging, log_level))
    handler = logging.FileHandler(log_file)
    handler.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class ModelWrapper:
    def __init__(self, args):
        self.args = args
        self.model_name = self.args.model_name
        self.model, self.tokenizer = load_model(self.args)


model_version = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3"
}


def main():
    parser = argparse.ArgumentParser(description="Run Self-Discover")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="gpt-oss-20b")
    parser.add_argument("--dataset_name", type=str, default="MATH500", help="T4D, MATH500, HotpotQA, StrategyQA")
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--data_dir", type=str, default="data")    

    parser.add_argument("--alpha_cap", type=float, default=5.0)
    parser.add_argument("--alpha_pos_cap", type=float, default=5.0)
    parser.add_argument("--alpha_neg_cap", type=float, default=2.0)
    parser.add_argument("--ppl_off", type=float, default=20.0)
    parser.add_argument("--ppl_on", type=float, default=150.0)
    parser.add_argument("--ppl_neg_on", type=float, default=5.0)

    parser.add_argument("--marker_gain_coef", type=float, default=1.00)  
    parser.add_argument("--marker_gain_cap", type=float, default=1.00)    
    parser.add_argument("--marker_dropout", type=float, default=0.00)   
    parser.add_argument("--marker_taper", action="store_true",  default=False)  

    parser.add_argument("--delta_gain_coef", type=float, default=0.07) 
    parser.add_argument("--delta_gain_cap", type=float, default=1.0) 

    parser.add_argument("--force_eager_attn", action="store_true")
    parser.add_argument("--disable_sdp_flash", action="store_true")

    args = parser.parse_args()
    
    log_dir = os.path.join(args.log_dir, args.dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{args.model_name}_{timestamp}.log")

    logger = setup_logging(log_file, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Generate Meta-prompt using {args.model_name} for {args.dataset_name}")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")


    output_dir = os.path.join(args.output_dir, args.dataset_name)
    output_path = os.path.join(output_dir, f"{args.model_name}.json")
    os.makedirs(output_dir, exist_ok=True)

    args.model_id = model_version[args.model_name]

    model = ModelWrapper(args)

    actions = ["SELECT", "ADAPT", "IMPLEMENT"]

    data_dir = os.path.join(args.data_dir, args.dataset_name)
    data_path = os.path.join(data_dir, f"{args.dataset_name}.json")
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    res = []
    for data in tqdm(dataset):
        runner = SelfDiscover(model, actions, data['question'], logger, args)
        runner()
        tmp = {
            "id": data['id'],
            "question": data['question'],
            "answer": data.get('answer'),
            "method": args.method,
            "selected_modules": runner.selected_modules,
            "using_rag": runner.using_rag,
            "adapted_modules": runner.adapted_modules,
            "reasoning_structure": runner.reasoning_structure,
            "spp_nll": runner.spp_nll,
            "spp_ppl": runner.spp_ppl,
            "alpha": runner.alpha,
            "used_scaled_embed": runner.used_scaled_embed,
            "m1_found": runner.m1_found,
            "m1_span_len": runner.m1_span_len,
            "scaled_tokens": runner.scaled_tokens,
            "marker_eff_gain": runner.marker_eff_gain,
            "delta_eff_gain": runner.delta_eff_gain,
        }
        res.append(tmp)

        with open(output_path, 'w') as f:
            json.dump(res, f, indent=4)
            logger.info(f"Results saved to {os.path.abspath(output_path)}")

    with open(output_path, 'w') as f:
        json.dump(res, f, indent=4)
        logger.info(f"All results saved to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
