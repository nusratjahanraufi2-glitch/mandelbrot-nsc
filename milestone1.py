import numpy as np

def main():
    x = np.linspace(-2, 1, 1024)
    y = np.linspace(-1.5, 1.5, 1024)

    X, Y = np.meshgrid(x, y)

    C = X + 1j * Y

    print("Shape:", C.shape)
    print("Type:", C.dtype)

if __name__ == "__main__":
    main()
