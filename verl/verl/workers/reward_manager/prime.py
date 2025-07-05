# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

import os
import json

async def single_compute_score(evaluation_func, completion, reference, task, extra_info, use_model_reward, executor, timeout=300.):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference, extra_info, use_model_reward, 'prime')  # Ensure synchronous
                ),
                timeout=timeout)
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_infos, use_model_reward, num_processes=64):
    scores = []
    metas = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, extra_info, use_model_reward, executor, timeout=600.)
            for completion, reference, task, extra_info in zip(completions, references, tasks, extra_infos)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
            metas.append('')
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
            metas.append('')
        else:
            scores.append(float(result[0][0]))
            metas.append(result[0][1])
    return scores, metas


class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, config, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.config = config

    def __call__(self, data: DataProto, global_steps):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_info = [data_item.non_tensor_batch.get('extra_info', None) for data_item in data]
        questions = [data_item.non_tensor_batch['question'] for data_item in data]

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores, metas = asyncio.run(
                parallel_compute_score_async(self.compute_score,
                                             sequences_str,
                                             ground_truth,
                                             data_sources,
                                             extra_infos=extra_info,
                                             use_model_reward=self.config.trainer.use_model_reward,
                                             num_processes=64))
        except asyncio.TimeoutError as e:
            print('Global timeout in reward computing! Setting all as 0.')
            scores = [0. for _ in range(len(sequences_str))]
            metas = ['' for _ in range(len(sequences_str))]

        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0. for _ in range(len(sequences_str))]
            metas = ['' for _ in range(len(sequences_str))]

        response_length = []
        total_score = []
        outlines = []
        for i in range(len(data)):
            data_source = data_sources[i]
            question = questions[i]
            ground_truth_i = ground_truth[i]
            response_length.append(valid_response_length[i].item())
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
            total_score.append(scores[i])
            outlines.append({'question': question, 'solution': ground_truth_i,'pred': sequences_str[i], 'score': scores[i], 'metadata': metas[i], 'global_steps': global_steps})

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        with open(os.path.join(self.config.trainer.default_local_dir, 'reward.jsonl'), 'a+', encoding='utf-8') as f: # save score
            for i in range(len(data)):
                data_i = outlines[i]
                f.write(json.dumps(data_i, ensure_ascii=False)+'\n')

        return reward_tensor
