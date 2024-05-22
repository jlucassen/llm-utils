from concurrent.futures import ThreadPoolExecutor, Future
from tqdm import tqdm
import time

class Interface:
    def __init__(self):
        self.queue = []

    def submit(self, func, *args):
        future = Future()
        self.queue.append((func, args, future))
        return future

    def go(self):
        with ThreadPoolExecutor(max_workers=1000) as executor:
            pbar = tqdm(total=len(self.queue))
            
            def wrapper(func, args, future):
                try:
                    result = func(*args)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                pbar.update(1)

            futures = []
            for func, args, future in self.queue:
                f = executor.submit(wrapper, func, args, future)
                futures.append(f)
            
            for f in futures:
                f.result()  # Wait for all futures to complete
            
            pbar.close()

def test(n):
    time.sleep(n % 5)
    return n % 10

interface = Interface()
out = [interface.submit(test, n) for n in range(1000)]
print(out[0])  # Should print a Future object

interface.go()

# After go() has been called and tasks are complete
print(out[0].result())  # Should print 0 (result of test(0))
print(out[0])
