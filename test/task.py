import time
import requests
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer


def call_score_api(text: str, return_all_scores: bool = False, url: str = "http://localhost:8000/score"):
    payload = {
        "text": text,
        "return_all_scores": return_all_scores
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return {"error": str(e)}


def call_response_api(prefix_prompt, max_tokens=1024, response_num=8, stop=[], temperature=0.8, url_list=["http://localhost:7999/generate"]):
    url_count = len(url_list)
    base_num = response_num // url_count
    remainder = response_num % url_count
    num_per_url = [base_num + (1 if i < remainder else 0) for i in range(url_count)]

    def send_request(url, payload):
        if num == 0:
            return {}  
        try:
            response = requests.post(url, json=payload, timeout=3600)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Request to {url} failed: {e}")
            return {"error": str(e)}

    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for url, num in zip(url_list, num_per_url):
            payload = {
                "prefix_prompt": prefix_prompt,
                "max_tokens": max_tokens,
                "response_num": num,
                "stop": stop,
                "temperature": temperature,
                "seed": secrets.randbelow(2**32),
            }
            futures.append(executor.submit(send_request, url, payload))
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    combined_responses = []
    for r in results:
        if 'results' in r:
            combined_responses.extend(r["results"])
    return combined_responses


class Infer_Task:
    def __init__(self, model_dir, score_api_url, response_api_url, branch=3, temperature=0.7, max_tokens=1024*32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.score_api_url = score_api_url
        self.response_api_url = response_api_url
        self.branch = branch
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self,question):
        time1 = time.time()
        messages = [{"role": "user", "content": question}]
        question = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = call_response_api(question, max_tokens=self.max_tokens, response_num=self.branch, temperature=self.temperature, url_list=self.response_api_url)
        outputs = [question + output for output in outputs]
        values = [call_score_api(y, return_all_scores=False, url=self.score_api_url)['value'] for y in outputs]
        time2 = time.time()
        
        assert len(outputs)==self.branch
        final_answer = {'content': question, 'time': time2-time1, 'solution_lst': outputs, 'score_lst': values}
        return final_answer, None
        