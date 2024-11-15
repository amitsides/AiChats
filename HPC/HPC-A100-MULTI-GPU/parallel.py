import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def process_tensor(rank, world_size, tensor):
    setup(rank, world_size)
    
    # Move tensor to GPU
    device = torch.device(f"cuda:{rank}")
    local_tensor = tensor.to(device)
    
    # Perform operations on the tensor
    result = local_tensor * 2  # Example operation
    
    # Synchronize results across GPUs
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    
    cleanup()
    return result

def main():
    world_size = torch.cuda.device_count()
    tensor = torch.randn(1000, 1000)  # Example tensor
    
    mp.spawn(process_tensor,
             args=(world_size, tensor),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()