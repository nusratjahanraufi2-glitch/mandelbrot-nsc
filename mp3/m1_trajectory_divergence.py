import numpy as np
import matplotlib.pyplot as plt

# Parameters from the slide
N = 512
MAX_ITER = 1000
TAU = 0.01

# Seahorse Valley region from the slide
x = np.linspace(-0.7530, -0.7490, N)
y = np.linspace(0.0990, 0.1030, N)

# Build float64 and float32 complex grids
C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
C32 = C64.astype(np.complex64)

# Initial trajectories
z32 = np.zeros_like(C32)
z64 = np.zeros_like(C64)

# Store first divergence iteration
# MAX_ITER means "did not diverge before max_iter"
diverge = np.full((N, N), MAX_ITER, dtype=np.int32)

# Active points are those still being tracked
active = np.ones((N, N), dtype=bool)

for k in range(MAX_ITER):
    if not active.any():
        break

    # Update only active points
    z32[active] = z32[active] ** 2 + C32[active]
    z64[active] = z64[active] ** 2 + C64[active]

    # Stop tracking points that already escaped
    escaped = active & (np.abs(z64) > 2.0)

    # Difference between float32 and float64 trajectories
    diff = np.abs(z32.astype(np.complex128) - z64)

    # Record first divergence iteration
    newly_diverged = active & (diff > TAU)
    diverge[newly_diverged] = k

    # Remove escaped and diverged points from further tracking
    active[escaped | newly_diverged] = False

# Fraction of pixels that diverged before max_iter
diverged_fraction = np.mean(diverge != MAX_ITER)

print(f"Fraction of pixels diverged before max iter: {diverged_fraction:.4f}")
print(f"Region: x=[-0.7530, -0.7490], y=[0.0990, 0.1030], tau={TAU}, max_iter={MAX_ITER}")

# For clearer visualization:
# show non-diverged pixels as 0 so background stays dark
display = diverge.copy()
display[display == MAX_ITER] = 0

plt.imshow(
    display,
    cmap="plasma",
    origin="lower",
    extent=[-0.7530, -0.7490, 0.0990, 0.1030]
)
plt.colorbar(label="First divergence iteration")
plt.title(f"Trajectory divergence (tau={TAU})")
plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")
plt.show()