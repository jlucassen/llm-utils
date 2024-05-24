from openai import OpenAI
import tiktoken
from anthropic import Anthropic
import google.generativeai as genai

from concurrent.futures import ThreadPoolExecutor, Future
from tqdm import tqdm

import os
import dotenv
import json


# TODO
# rate limiting
# rate limiting for multiple APIs

class LLM_Interface:
    def __init__(self):
        self.queue = []
        dotenv.load_dotenv()
        self.openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

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
        self.queue = []

    def _tokenize_queue(self):
        tokens_per_query = []
        tokenizers = {}
        for func, kwargs, future in tqdm(self.queue):
            if func.__name__ == 'queue_openai_func':
                if kwargs['model'] not in tokenizers:
                    tokenizers[kwargs['model']] = tiktoken.encoding_for_model(kwargs['model'])
                tokens_per_query.append((len(tokenizers[kwargs['model']].encode(''.join(kwargs['messages'][0].values()))), kwargs['model']))
            elif func.__name__ == 'queue_anthropic_func':
                tokens_per_query.append((int(len(''.join(kwargs['messages'][0].values()))/4), kwargs['model']))
            elif func.__name__ == 'queue_google_func':
                tokens_per_query.append((kwargs['model'].count_tokens(kwargs['prompt']).total_tokens, kwargs['model'].model_name))
        return tokens_per_query
    
    def count_tokens(self):
        return sum([x[0] for x in self._tokenize_queue()])
    
    def cost_estimate(self, output = 0):
        prices = json.load(open('prices.json'))
        tokenized_queue = self._tokenize_queue()
        return sum([(tokenized_queue[i][0])*prices[tokenized_queue[i][1]][0] + output*prices[tokenized_queue[i][1]][1] for i in range(len(tokenized_queue))])/10**6

    def queue_openai(self, model, prompt, kwargs={}):
        def queue_openai_func(messages, model, kwargs): # wrapper function to extract text at the end
            result = self.openai.chat.completions.create(messages=messages, model=model, **kwargs)
            return result.choices[0].message.content
        
        prompt = [{"role": "user", "content": prompt}]
        kwargs = {'messages':prompt, 'model':model, 'kwargs':kwargs}
        future = Future()
        self.queue.append((queue_openai_func, kwargs, future))
        return future
    
    def queue_anthropic(self, model, prompt, max_tokens, kwargs={}):
        def queue_anthropic_func(messages, model, max_tokens, kwargs): # wrapper function to extract text at the end
            result = self.anthropic.messages.create(messages=messages, model=model, max_tokens=max_tokens, **kwargs)
            return result.content[0].text
        
        prompt = [{"role": "user", "content": prompt}]
        kwargs = {'messages':prompt, 'model':model, 'max_tokens':max_tokens, 'kwargs':kwargs}
        future = Future()
        self.queue.append((queue_anthropic_func, kwargs, future))
        return future

    def queue_google(self, model, prompt, kwargs={}):
        def queue_google_func(prompt, model, kwargs): # wrapper function to extract text at the end
            return model.generate_content(prompt, **kwargs).text
    
        kwargs = {'prompt':prompt, 'model':model, 'kwargs':kwargs}
        future = Future()
        self.queue.append((queue_google_func, kwargs, future))
        return future

    def queue_function(self, func, kwargs={}):
        future = Future()
        self.queue.append((func, kwargs, future))
        return future

def main():
    llm = LLM_Interface()
    x = llm.queue_openai("gpt-3.5-turbo", "Who are you?", {"max_tokens": 10})
    y = llm.queue_anthropic('claude-3-opus-20240229', 'Who are you?', 20, {'system': 'Respond only in Yoda-speak.'})
    z = llm.queue_google(genai.GenerativeModel('gemini-pro'), 'Who are you?', {'safety_settings':{'HARASSMENT':'block_none'}})
    a = llm.queue_function(lambda x: x**2, {'x': 5})
    print(llm.cost_estimate(0))
    print(llm.count_tokens())
    print(llm.cost_estimate(100))
    llm.go()
    print(x.result())
    print(y.result())
    print(z.result())
    print(a.result())

if __name__ == "__main__":
    main()