import torch
import time

# Check if CUDA is available
if not torch.cuda.is_available():
    print('CUDA not available')
    exit()

print(f'GPU: {torch.cuda.get_device_name(0)}')
print()

# Test different sizes
configs = [
    (30000, 32000, 32),   # typical: 30k tokens, 32k vocab, k=32
    (30000, 32000, 64),   # k=64
    (30000, 32000, 512),  # k=512
    (8192, 32000, 32),    # smaller batch
    (8192, 32000, 512),   # smaller batch, larger k
]

for total_tokens, vocab_size, k in configs:
    logits = torch.randn(total_tokens, vocab_size, device='cuda', dtype=torch.bfloat16)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(3):
        _, indices = logits.topk(k, dim=-1)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    n_iters = 20
    for _ in range(n_iters):
        _, indices = logits.topk(k, dim=-1)
    torch.cuda.synchronize()

    elapsed = (time.time() - start) / n_iters * 1000

    # Also measure gather
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        gathered = logits.gather(-1, indices)
    torch.cuda.synchronize()
    gather_elapsed = (time.time() - start) / n_iters * 1000

    print(f'Shape ({total_tokens}, {vocab_size}), k={k}:')
    print(f'  topk:   {elapsed:.2f} ms')
    print(f'  gather: {gather_elapsed:.2f} ms')
    print(f'  total:  {elapsed + gather_elapsed:.2f} ms')
    print()

    del logits, indices
    torch.cuda.empty_cache()
