import os
import time
import torch
import torch.nn as nn
import numpy as np
import asyncio
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


parser = argparse.ArgumentParser(description="Run reward model API")
parser.add_argument("--model_path", type=str, default='path/to/train_models')
parser.add_argument("--score_model_dim", type=int, default=1536)
parser.add_argument("--lang", type=str, choices=['en', 'zh'])
parser.add_argument("--ip", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default=8000)
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
    text: str
    return_all_scores: bool = False

# ============ 推理主逻辑 ============

def get_all_key_ids(tokenizer, target_chars=['\n','。']):
    if args.lang=='zh':
        vocab = tokenizer.get_vocab()
        target_token_ids = []
        for token in vocab.keys():
            try:
                token_id = tokenizer.convert_tokens_to_ids([token])[0]
                decoded = tokenizer.decode([token_id])
                for target_char in target_chars:
                    if target_char in decoded and len(decoded.replace(target_char,''))<=1 and len(decoded)<=3:
                        target_token_ids.append(token_id)
                        break
            except:
                continue
    else:
        target_token_ids = tokenizer.encode('.\n\n', add_special_tokens=False)
    return target_token_ids

def get_score_mask(x, target_token_ids, think_token_id): # (1,L) tensor, list, int
    mask = torch.isin(x, torch.tensor(target_token_ids, device=x.device))

    mask_shifted = torch.nn.functional.pad(mask[:, :-1], (1, 0), value=False)
    mask_first_only = mask & ~mask_shifted

    think_pos = (x == think_token_id)
    think_next_mask = torch.nn.functional.pad(think_pos[:, :-1], (1, 0), value=False)  
    final_mask = mask_first_only & (~think_next_mask)

    final_mask = torch.nn.functional.pad(final_mask[:, 1:], (0, 1), value=False)

    return final_mask

def geometric_mean(tensor, dim=None, eps=1e-10):
    if tensor.shape[0] == 0:
        return torch.tensor(0, dtype=tensor.dtype, device=tensor.device)
    tensor = torch.clamp(tensor, min=eps)  
    log_tensor = torch.log(tensor)
    mean_log = torch.mean(log_tensor, dim=dim)
    return torch.exp(mean_log)

class ScoreModel:
    def __init__(self, model_path, score_model_path, score_model_dim=1536):
        self.score_model_dim = score_model_dim
        logger.info("Loading LLM and tokenizer...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='auto',
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.think_start_id = self.tokenizer.encode('<think>', add_special_tokens=False)[0]
        self.think_end_id = self.tokenizer.encode('</think>', add_special_tokens=False)[0]
        self.score_id = get_all_key_ids(self.tokenizer)
        self.score_model = self.__init_score_model(score_model_path)
        logger.info("Models loaded.")

    def __init_score_model(self, score_model_path):
        model = nn.Sequential(
            nn.Linear(self.score_model_dim, self.score_model_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.score_model_dim * 2, 1)
        )
        state_dict = torch.load(score_model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval().to(self.llm.device).to(self.llm.dtype)
        return model

    def __call__(self, text, return_all_scores=False):
        with torch.inference_mode():
            logger.info("Running inference...")
            new_input = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.llm.device)
            # logger.info(f"Input: {text}")
            output = self.llm(**new_input, output_hidden_states=True)
            feature = output.hidden_states[-2]
            input_ids = new_input.input_ids

            score_mask = get_score_mask(input_ids, self.score_id, self.think_start_id)
            start_pos = (input_ids[0] == self.think_start_id).nonzero(as_tuple=True)[0]
            start_pos = start_pos.item() if start_pos.numel() == 1 else 0

            end_pos = (input_ids[0] == self.think_end_id).nonzero(as_tuple=True)[0]
            end_pos = end_pos.item() if end_pos.numel() == 1 else input_ids[0].shape[0]

            between_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            between_mask[:, start_pos + 1:end_pos] = True
            between_mask &= score_mask

            indexs = np.where(between_mask.cpu().numpy() == True)[1]
            pred_score = torch.sigmoid(self.score_model(feature[between_mask]).float())
            value = geometric_mean(pred_score).detach().cpu().item()
            if value==0:
                logger.info(f"{pred_score}")
                logger.info(f"{text}")
            logger.info(f"Score: {value}")
            del new_input, output, feature
            torch.cuda.empty_cache()

            if return_all_scores:
                all_scores = pred_score.detach().cpu().numpy().flatten().tolist()
                return value, all_scores, input_ids.detach().cpu()[0].tolist(), indexs.tolist()
            else:
                return value

# ============ 加载模型 ============
score_model = ScoreModel(
    model_path=args.model_path,             
    score_model_path=os.path.join(args.model_path, 'score_module.pt'),   
    score_model_dim=args.score_model_dim,    
)

# ============ 接口 + 队列调度 ============

@app.post("/score")
async def get_score(input: TextInput):
    loop = asyncio.get_event_loop()
    response_future = loop.create_future()

    async def inference_task():
        max_retry = 3
        for attempt in range(max_retry):
            try:
                logger.info(f"Inference attempt {attempt + 1}")
                result = score_model(input.text, return_all_scores=input.return_all_scores)
                if input.return_all_scores:
                    response_future.set_result({
                        "value": result[0],
                        "all_scores": result[1],
                        "token_ids": result[2],
                        "score_positions": result[3]
                    })
                else:
                    response_future.set_result({"value": result})
                return
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(f"CUDA OOM on attempt {attempt + 1}")
                    torch.cuda.empty_cache()
                    await asyncio.sleep(0.5)
                    continue
                else:
                    logger.exception("Runtime error")
                    response_future.set_result({"error": str(e)})
                    return
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