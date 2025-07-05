import os
from task import Infer_Task
import pandas as pd
import json
import argparse
import traceback
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Run inference on model with prompts from a jsonl file")
parser.add_argument("--task", type=str, default='aime24', help="the target benchmark (e.g., aime24, aime25)")
parser.add_argument("--input_file", type=str, default='data/aime.parquet', help="the data file of the target benchmark")
parser.add_argument("--output_file", type=str, default="./output.jsonl", help="the save path for the test result")
parser.add_argument("--branch", type=int, default=16, help="the number of candidates for each inference (recommended: 2x the number of policy model APIs)")
parser.add_argument("--n_samples", type=int, default=64, help="test n times for each question")
parser.add_argument("--score_api_url", type=str, default="", help="the URL of the reward model API")
parser.add_argument("--response_api_url", type=str, default="", help="the URL list of policy model API, use ',' to split multiple urls")
parser.add_argument("--model_dir", type=str, default='path/to/train_models', help="the location of the policy model")
parser.add_argument("--total_parts", type=int, default=1)
parser.add_argument("--part", type=int, default=0)
args = parser.parse_args()
args.response_api_url = args.response_api_url.split(',')


def read_file_to_list(file_path):
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    elif file_path.endswith(".jsonl"):
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError("Unsupported file format.")
    return df.to_dict(orient="records")


def parse_data(data, dataset='aime24', part=0, total_parts=2):
    if dataset in ['aime24', 'aime25']:
        return [{'prompt': item['prompt'], 'answer': item['answer']}
            for item in data[part::total_parts]]
    elif dataset=='ceval':
        return [{'prompt': item['model_input'], 'answer': item['answer']}
            for item in data[part::total_parts]]
    else:
        raise ValueError("Unsupported dataset.")


if __name__ == "__main__":
    print(f"Processing part {args.part}...")
    data = read_file_to_list(args.input_file)

    data = parse_data(data, dataset=args.task, part=args.part, total_parts=args.total_parts)

    data = data*args.n_samples
    if os.path.exists(args.output_file):
        processed_data = read_file_to_list(args.output_file)
        data = data[len(processed_data):]
    # print(len(data))
    model_dir = args.model_dir
    score_api_url = args.score_api_url
    response_api_url = args.response_api_url
    task = Infer_Task(model_dir=model_dir, score_api_url=score_api_url, response_api_url=response_api_url, branch=args.branch, max_tokens=1024*32, temperature=0.6)
    max_retries = 3
    with open(args.output_file, 'a') as f:
        for item in tqdm(data): 
            question = item['prompt']
            success = False
            
            for attempt in range(max_retries):
                try:
                    output = task.run(question)[0]
                    item['output'] = output
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    success = True
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for question: {question}")
                    print(e)
                    traceback.print_exc()
            assert success
        
    


