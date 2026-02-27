import numpy as np
import matplotlib.pyplot as plt
from mandelbrot_dtype import mandelbrot_kernel_f32, mandelbrot_kernel_f64, quantize_to_float16_then_float32

# Zoom region (Seahorse Valley) to reveal precision artefacts
xmin, xmax = -0.8, -0.7
ymin, ymax = 0.05, 0.15
width = height = 1024
max_iter = 500

x32 = np.linspace(xmin, xmax, width).astype(np.float32)
y32 = np.linspace(ymin, ymax, height).astype(np.float32)
x16q = quantize_to_float16_then_float32(x32)
y16q = quantize_to_float16_then_float32(y32)

x64 = np.linspace(xmin, xmax, width).astype(np.float64)
y64 = np.linspace(ymin, ymax, height).astype(np.float64)

r16 = mandelbrot_kernel_f32(x16q, y16q, max_iter)  # float16 precision simulated
r32 = mandelbrot_kernel_f32(x32, y32, max_iter)
r64 = mandelbrot_kernel_f64(x64, y64, max_iter)

print("Max diff float32 vs float64:", np.abs(r32 - r64).max())
print("Max diff float16-quantized vs float64:", np.abs(r16 - r64).max())

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, title in zip(
    axes,
    [r16, r32, r64],
    ["float16 precision (quantized)", "float32", "float64 (ref)"]
):
    ax.imshow(img, cmap="hot")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("week3_precision_comparison.png", dpi=150)
plt.show()