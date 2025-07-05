import json
import re
from glob import glob
import argparse
import numpy as np
import prime_math
import random


parser = argparse.ArgumentParser(description="Compute the metric of MetaStone-s1")
parser.add_argument("--task", type=str, default='aime24', help="the target benchmark (e.g., aime24, aime25)")
parser.add_argument("--N", type=int, default=8, help="the number of candidates for evaultation, 2/8/32 for low/medium/high")
parser.add_argument("--result_paths", type=str, default='results/aime24_*.jsonl', help="the location of result file")
args = parser.parse_args()

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

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


def get_res(gen, gt, dataset):
    if isinstance(gt, dict):
        gt = gt['target']
    if dataset.startswith('aime'):
        score = prime_math.compute_score(gen, gt)
        return int(score[0] and score[1])
    elif dataset=='ceval':
        ANSWER_PATTERN = r'(?i)(?:答案|answer)\s*[:：]\s*([A-Da-d])\.?'
        match = re.search(ANSWER_PATTERN, gen)
        if match:
            res = match.group(1).lower()
        else:
            res = ''
        score = int(res.strip().lower() == gt.strip().lower())
        return score
    else:
        raise ValueError("Unsupported dataset.")

def compute_bon_results(datax, task):
    correct_predictions = 0
    total_predictions = 0
    for data in datax:
        gens = data['solutions']
        gold = data['answer']
        scores = data['scores']
        correctness = []
        for gen in gens:
            res = get_res(gen, gold, task) 
            correctness.append(res)

        max_score_index = scores.index(max(scores))
        if correctness[max_score_index]: 
            correct_predictions += 1 

        total_predictions += 1
                
    avg_accuracy = correct_predictions / total_predictions if total_predictions else 0

    return avg_accuracy

if __name__ == "__main__":
    paths = glob(args.result_paths)

    data = []
    for path in paths:
        data += load_jsonl(path)
    grouped_jobs = {}
    for item in data:
        question = item['prompt']
        if question not in grouped_jobs:
            grouped_jobs[question] = {'answer':item['answer'], 'solutions':item["output"]['solution_lst'], 'scores':item["output"]['score_lst']}
        else:
            grouped_jobs[question]['solutions'] += item["output"]['solution_lst']
            grouped_jobs[question]['scores'] += item["output"]['score_lst']

    solution_nums = []
    for item in grouped_jobs.values():
        solution_nums.append(len(item['solutions']))
    K = min(solution_nums)//args.N
    print(f'Test {K} times')
    assert K>=1, min_solutions

    k_groups = [[] for _ in range(K)]
    for prompt, job_dict in grouped_jobs.items():
        group_size = args.N
        for i in range(K):
            grouped_job = {}
            grouped_job['answer'] = job_dict['answer']
            grouped_job['solutions'] = job_dict['solutions'][i * group_size:(i + 1) * group_size]
            grouped_job['scores'] = job_dict['scores'][i * group_size:(i + 1) * group_size]
            k_groups[i].append(grouped_job)

    accs = []
    for k_group in k_groups:
        acc = compute_bon_results(k_group, args.task)
        accs.append(acc)

    accs_mean =  np.mean(accs)

    print(accs_mean)
