import torch

sizes = [1_000_000, 10_000_000, 100_000_000]
bandwidth_bytes_per_sec = 320e9  # T4 theoretical peak

print(f"{'n':>12} | {'measured (min)':>16} | {'predicted floor':>16} | {'ratio':>6}")
print("-" * 62)

for n in sizes:
    a = torch.rand(n, device='cuda')
    b = torch.rand(n, device='cuda')

    # warm-up
    c = a + b
    torch.cuda.synchronize()

    times = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        c = a + b
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    min_ms = min(times)
    min_us = min_ms * 1000

    total_bytes = n * 4 * 3  # read a, read b, write c, 4 bytes/float32
    predicted_us = (total_bytes / bandwidth_bytes_per_sec) * 1e6
    ratio = min_us / predicted_us

    print(f"{n:>12,} | {min_us:>13.2f} us | {predicted_us:>13.2f} us | {ratio:>5.2f}x")
