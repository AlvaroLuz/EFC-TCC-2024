import time
import threading

class Log():
    def __init__(self):
        #definir modelo com seus parametros e definir o dataset
        self.start_time = time.time()
        self.timer_thread()
    
    def log(self, message):
        current_time = time.time() - self.start_time
        print(f"\r[{current_time:.2f}s] Log: {message}")
    
    def timer_thread(self):
        self.running = True  # Add a flag to control the thread
        def log_thread():
            while self.running:  # Check the flag in the loop condition
                current_time = time.time() - self.start_time
                print(f"\r[{current_time:.2f}s] Running...", end="")
                time.sleep(1)
        self.thread = threading.Thread(target=log_thread)
        self.thread.start()

    def func_time(self, what, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            total_time = end_time - start_time
            self.log(f"{what} time: {total_time:.2f}s")
            return result
        return wrapper

    def __del__(self):
        self.running = False  # Set the flag to stop the thread
        self.thread.join()  # Wait for the thread to finish
