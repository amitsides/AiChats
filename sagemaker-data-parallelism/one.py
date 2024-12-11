# Parameter Server Architecture Example
import sagemaker
from sagemaker.pytorch import PyTorch

def parameter_server_training():
    # Configure parameter server distribution
    distribution = {
        'parameter_server': {
            'enabled': True
        }
    }
    
    pytorch_estimator = PyTorch(
        entry_point='train.py',
        role='SageMakerRole',
        instance_count=4,  # 1 parameter server + 3 workers
        instance_type='ml.p3.2xlarge',
        framework_version='1.8.1',
        py_version='py3',
        distribution=distribution
    )

    # Training script (train.py)
    def train():
        import torch.distributed as dist
        
        # Initialize process group
        dist.init_process_group(backend='gloo')
        
        # Load and shard data across workers
        dataset = load_dataset()
        sampler = DistributedSampler(dataset)
        
        for epoch in range(num_epochs):
            for batch in DataLoader(dataset, sampler=sampler):
                # Forward pass
                outputs = model(batch)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Parameter server handles gradient aggregation
                optimizer.step()
                
                # Parameters automatically synced via parameter server

# All-Reduce Architecture Example
def allreduce_training():
    # Configure All-Reduce distribution
    distribution = {
        'torch_distributed': {
            'enabled': True
        }
    }
    
    pytorch_estimator = PyTorch(
        entry_point='train.py',
        role='SageMakerRole',
        instance_count=4,  # All instances are workers
        instance_type='ml.p3.2xlarge',
        framework_version='1.8.1',
        py_version='py3',
        distribution=distribution
    )

    # Training script (train.py)
    def train():
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        
        # Initialize process group with NCCL backend
        dist.init_process_group(backend='nccl')
        
        # Wrap model in DistributedDataParallel
        model = DistributedDataParallel(model)
        
        # Load and shard data across workers
        dataset = load_dataset()
        sampler = DistributedSampler(dataset)
        
        for epoch in range(num_epochs):
            for batch in DataLoader(dataset, sampler=sampler):
                # Forward pass
                outputs = model(batch)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # All-Reduce automatically handles gradient synchronization
                optimizer.step()
                
                # DDP handles parameter synchronization via ring all-reduce

# Helper functions for both architectures
def load_dataset():
    # Load your dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = YourDataset(
        root='data',
        transform=transform
    )
    return dataset

def calculate_batch_size(num_gpus):
    # Scale batch size with number of GPUs
    base_batch_size = 32
    global_batch_size = base_batch_size * num_gpus
    return global_batch_size