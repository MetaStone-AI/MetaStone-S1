import os
import time
import torch
import torch.nn as nn
import numpy as np
import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import copy


parser = argparse.ArgumentParser(description="Run policy model API")
parser.add_argument("--model_path", type=str, default='path/to/train_models')
parser.add_argument("--ip", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default=7999)
args = parser.parse_args()

# ============ 初始化日志 ============

# LOG_DIR = "log_score"
# os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        # logging.FileHandler(f"{LOG_DIR}/api.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============ 定义FastAPI和队列 ============
app = FastAPI()
MAX_CONCURRENT_TASKS = 1
task_queue = asyncio.Queue()


@app.on_event("startup")
async def start_worker():
    for _ in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(queue_worker())

async def queue_worker():
    while True:
        task = await task_queue.get()
        try:
            await task()
        except Exception as e:
            logger.error(f"Queue worker error: {str(e)}")
        task_queue.task_done()

# ============ 输入格式 ============
class TextInput(BaseModel):
    prefix_prompt: str
    max_tokens: int = 1024
    response_num: int = 8
    stop: list = None
    temperature: float = 0.8
    seed: int = None

# ============ 推理主逻辑 ============
class QwenVLLM:
    def __init__(self, model, temperature=0.8, top_p=0.95, max_tokens=32768, repetition_penalty=1.05, gpu_memory_utilization=0.8):
        print(f"Initializing vLLM on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
        gpu_num = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
        self.vllm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, dtype=torch.bfloat16, tensor_parallel_size=gpu_num)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "left"
        self.temperature = temperature
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
    
    def __call__(self, prefix_prompt, max_tokens=1024, response_num=8, stop=None, temperature=None, seed=None):
        current_samling_params = copy.deepcopy(self.sampling_params)
        current_samling_params.max_tokens = max_tokens
        current_samling_params.stop = stop if stop else [] 
        current_samling_params.include_stop_str_in_output = True
        current_samling_params.n = response_num
        current_samling_params.best_of = response_num
        if temperature:
            current_samling_params.temperature = temperature
        if seed:
            current_samling_params.seed = seed
        current_samling_params.return_hidden_states=False
        # outputs = self.vllm.generate([prefix_prompt]*response_num, current_samling_params, use_tqdm=False)
        # generated_texts = [output.outputs[0].text for output in outputs]
        outputs = self.vllm.generate(prefix_prompt, current_samling_params, use_tqdm=False)
        generated_texts = [res.text for res in outputs[0].outputs]
        return generated_texts


# ============ 加载模型 ============
model_vllm = QwenVLLM(
    model=args.model_path,
    temperature=0.8,
    top_p=0.95,
    max_tokens=32768,
    repetition_penalty=1.05,
    gpu_memory_utilization=0.7
)

# ============ 接口 + 队列调度 ============

@app.post("/generate")
async def get_generate(input: TextInput):
    loop = asyncio.get_event_loop()
    response_future = loop.create_future()

    async def inference_task():
        max_retry = 3
        for attempt in range(max_retry):
            try:
                logger.info(f"Inference attempt {attempt + 1}")
                result = model_vllm(input.prefix_prompt, max_tokens=input.max_tokens, response_num=input.response_num, stop=input.stop, temperature=input.temperature, seed=input.seed)
                response_future.set_result({
                    "results":result,
                })
                return
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(f"CUDA OOM on attempt {attempt + 1}")
                    torch.cuda.empty_cache()
                    await asyncio.sleep(0.5)
                    continue
                else:
                    logger.exception("Runtime error")
                    torch.cuda.empty_cache()
                    await asyncio.sleep(0.5)
                    continue
            except Exception as e:
                logger.exception("Unexpected error")
                response_future.set_result({"error": str(e)})
                return

        logger.error("Max retry exceeded")
        response_future.set_result({
            "error": "CUDA out of memory even after retrying. Please try again later."
        })

    await task_queue.put(inference_task)
    return await response_future


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.ip, port=args.port, log_level="info")