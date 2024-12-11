import ray
import numpy as np
from typing import List, Union, Optional

@ray.remote
class ParameterWorker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.buffer = None
        
    def set_data(self, data):
        self.buffer = data
        return True
        
    def get_data(self):
        return self.buffer
        
    async def reduce_scatter(self, other_workers: List['ParameterWorker']):
        chunk_size = len(self.buffer) // self.world_size
        chunks = np.array_split(self.buffer, self.world_size)
        
        # Each worker is responsible for reducing its corresponding chunk
        for i in range(self.world_size):
            if i == self.rank:
                # Collect corresponding chunks from all workers
                chunks_to_reduce = []
                for worker in other_workers:
                    worker_chunks = ray.get(worker.get_data.remote())
                    worker_chunks = np.array_split(worker_chunks, self.world_size)
                    chunks_to_reduce.append(worker_chunks[i])
                
                # Perform reduction
                reduced_chunk = np.sum(chunks_to_reduce, axis=0)
                chunks[i] = reduced_chunk
        
        self.buffer = np.concatenate(chunks)
        return True
        
    async def all_gather(self, other_workers: List['ParameterWorker']):
        gathered_chunks = []
        chunk_size = len(self.buffer) // self.world_size
        
        # Each worker broadcasts its chunk to all other workers
        for i in range(self.world_size):
            source_worker = other_workers[i]
            source_data = ray.get(source_worker.get_data.remote())
            gathered_chunks.append(source_data)
            
        self.buffer = np.concatenate(gathered_chunks)
        return True

def allreduce(data: np.ndarray, num_workers: Optional[int] = None) -> np.ndarray:
    """
    Implements allreduce using Ray actors for distributed data aggregation.
    
    Args:
        data: numpy array to be reduced
        num_workers: number of workers to use (default: number of CPUs)
        
    Returns:
        Reduced numpy array
    """
    if not ray.is_initialized():
        ray.init()
    
    if num_workers is None:
        num_workers = ray.cluster_resources()['CPU']
    
    # Split data among workers
    data_splits = np.array_split(data, num_workers)
    
    # Create parameter workers
    workers = [ParameterWorker.remote(rank=i, world_size=num_workers) 
              for i in range(num_workers)]
    
    # Distribute data to workers
    setup_futures = [workers[i].set_data.remote(data_splits[i]) 
                    for i in range(num_workers)]
    ray.get(setup_futures)
    
    # Perform reduce-scatter
    scatter_futures = [worker.reduce_scatter.remote(workers) 
                      for worker in workers]
    ray.get(scatter_futures)
    
    # Perform all-gather
    gather_futures = [worker.all_gather.remote(workers) 
                     for worker in workers]
    ray.get(gather_futures)
    
    # Collect results from first worker
    result = ray.get(workers[0].get_data.remote())
    
    return result

# Example usage:
if __name__ == "__main__":
    ray.init()
    
    # Create sample data
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    
    # Perform allreduce
    result = allreduce(data, num_workers=2)
    print(f"Input data: {data}")
    print(f"Reduced result: {result}")
    
    ray.shutdown()