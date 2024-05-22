from concurrent.futures import ThreadPoolExecutor, Future
from tqdm import tqdm
import os

import dotenv
from openai import OpenAI
from anthropic import Anthropic

# TODO
# set up calls for multiple APIs, API keys
# cost estimation
# error handling
# cost threshold confirm message

class LLM_Interface:
    def __init__(self):
        self.queue = []
        dotenv.load_dotenv()
        self.openai = OpenAI()
        self.anthropic = Anthropic()

    def go(self):
        with ThreadPoolExecutor(max_workers=max(1, min(len(self.queue), 1000))) as executor:
            pbar = tqdm(total=len(self.queue))
            
            def wrapper(func, kwargs, future): # wrapper function to catch exceptions
                try:
                    result = func(**kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                pbar.update(1)

            futures = []
            for func, kwargs, future in self.queue:
                f = executor.submit(wrapper, func, kwargs, future)
                futures.append(f)
            
            for f in futures:
                f.result()  # Wait for all futures to complete
            
            pbar.close()

    def queue_openai(self, model, prompt, kwargs):
        def queue_openai_func(messages, model, kwargs): # wrapper function to extract text at the end
            result = self.openai.chat.completions.create(messages=messages, model=model, **kwargs)
            return result.choices[0].message.content
        
        prompt = [{"role": "user", "content": prompt}]
        kwargs = {'messages':prompt, 'model':model, 'kwargs':kwargs}
        future = Future()
        self.queue.append((queue_openai_func, kwargs, future))
        return future
    
    def queue_anthropic(self, model, prompt, max_tokens, kwargs):
        def queue_anthropic_func(messages, model, max_tokens, kwargs): # wrapper function to extract text at the end
            result = self.anthropic.messages.create(messages=messages, model=model, max_tokens=max_tokens, **kwargs)
            return result.content[0].text
        
        prompt = [{"role": "user", "content": prompt}]
        kwargs = {'messages':prompt, 'model':model, 'max_tokens':max_tokens, 'kwargs':kwargs}
        future = Future()
        self.queue.append((queue_anthropic_func, kwargs, future))
        return future

llm = LLM_Interface()
x = llm.queue_openai("gpt-3.5-turbo", "Once upon a time", {"max_tokens": 10})
y = llm.queue_anthropic('claude-3-opus-20240229', 'how are you?', 20, {'system': 'Respond only in Yoda-speak.'})
llm.go()
print(x.result())
print(y.result())