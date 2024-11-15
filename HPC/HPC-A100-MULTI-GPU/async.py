import asyncio
import torch

async def process_on_gpu(tensor, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")
    local_tensor = tensor.to(device)
    # Simulate some processing
    await asyncio.sleep(1)  
    return local_tensor * 2

async def main():
    tensor = torch.randn(1000, 1000)
    num_gpus = torch.cuda.device_count()
    
    tasks = [process_on_gpu(tensor, i) for i in range(num_gpus)]
    results = await asyncio.gather(*tasks)
    
    # Combine results if needed
    final_result = torch.stack(results).sum(dim=0)
    print(final_result.shape)

if __name__ == "__main__":
    asyncio.run(main())