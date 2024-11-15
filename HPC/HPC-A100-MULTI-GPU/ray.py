import ray
from ray.util.queue import Queue
from functools import wraps
import torch

# Initialize Ray with GPU support
ray.init(num_gpus=10)

# Create a queue for tasks
task_queue = Queue()

# Decorator to submit tasks to the queue
def gpu_task(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        task_queue.put((func, args, kwargs))
        return None
    return wrapper

# Worker function to process tasks on available GPUs
@ray.remote(num_gpus=1)
def process_task(func, *args, **kwargs):
    with torch.cuda.device(ray.get_gpu_ids()[0]):
        return func(*args, **kwargs)

# Main loop to consume tasks from the queue
while True:
    task, args, kwargs = task_queue.get()
    result = ray.get(process_task.remote(task, *args, **kwargs))
    print(f"Task completed: {task.__name__}")