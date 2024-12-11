import numpy as np
from mpi4py import MPI

def allreduce_ring(local_data, comm=MPI.COMM_WORLD, op=MPI.SUM):
    """
    Implements ring-based allreduce algorithm for distributed data aggregation.
    
    Args:
        local_data: numpy array containing local data to be reduced
        comm: MPI communicator (default: COMM_WORLD)
        op: Reduction operation (default: SUM)
    
    Returns:
        numpy array containing the reduced result
    """
    size = comm.Get_size()  # Number of processes
    rank = comm.Get_rank()  # Current process rank
    
    # Split data into equal chunks
    chunk_size = len(local_data) // size
    chunks = [local_data[i:i + chunk_size] for i in range(0, len(local_data), chunk_size)]
    
    # Initialize receive buffer
    recv_chunk = np.zeros_like(chunks[0])
    result = local_data.copy()
    
    # Ring reduce-scatter
    for i in range(size - 1):
        # Calculate source and destination ranks
        src = (rank - 1) % size
        dst = (rank + 1) % size
        
        # Send to next process and receive from previous process
        send_chunk = chunks[(rank - i) % size]
        comm.Sendrecv(send_chunk, dst, recvbuf=recv_chunk, source=src)
        
        # Perform local reduction
        if op == MPI.SUM:
            chunks[(rank - i - 1) % size] += recv_chunk
        elif op == MPI.PROD:
            chunks[(rank - i - 1) % size] *= recv_chunk
        elif op == MPI.MAX:
            chunks[(rank - i - 1) % size] = np.maximum(chunks[(rank - i - 1) % size], recv_chunk)
        elif op == MPI.MIN:
            chunks[(rank - i - 1) % size] = np.minimum(chunks[(rank - i - 1) % size], recv_chunk)
    
    # Ring allgather
    for i in range(size - 1):
        # Calculate source and destination ranks
        src = (rank - 1) % size
        dst = (rank + 1) % size
        
        # Send to next process and receive from previous process
        send_chunk = chunks[(rank - i) % size]
        comm.Sendrecv(send_chunk, dst, recvbuf=recv_chunk, source=src)
        
        # Copy received chunk to appropriate position
        chunks[(rank - i - 1) % size] = recv_chunk
    
    # Reconstruct final result
    result = np.concatenate(chunks)
    return result

def allreduce(data, comm=MPI.COMM_WORLD, op=MPI.SUM):
    """
    Wrapper function that handles both scalar and array inputs for allreduce.
    
    Args:
        data: scalar or numpy array to be reduced
        comm: MPI communicator
        op: reduction operation
    
    Returns:
        Reduced result
    """
    if isinstance(data, (int, float)):
        # Handle scalar values
        return comm.allreduce(data, op=op)
    else:
        # Handle numpy arrays
        return allreduce_ring(data, comm=comm, op=op)