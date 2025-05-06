import math
import numpy as np
from PIL import Image
import argparse

def fft(x):
    """
    Computes the 1D FFT of an array using
    via the Cooley-Tukey algorithm.
    Length of the array must be a power of 2.
    """

    N = x.shape[0]

    if N <= 1:
        return x.astype(np.complex128)

    if N % 2 != 0:
        raise ValueError("Size of x must be a power of 2")

    X_even = fft(x[::2])
    X_odd  = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)

    return np.concatenate([
        X_even + factor[:N//2] * X_odd,
        X_even + factor[N//2:] * X_odd
    ])

def ifft(X):
    """
    Computes the inverse 1D FFT or an array.
    """

    x_conj = np.conjugate(X)

    return np.conjugate(fft(x_conj)) / X.shape[0]

def fft2d(a):
    """
    Computes the 2D FFT by applying the 1D FFT
    on the rows and columns.
    """

    temp = np.array([fft(row) for row in a])
    temp = np.array([fft(col) for col in temp.T]).T

    return temp

def ifft2d(A):
    """
    Computes the inverse 2D FFT by applying
    the 1D IFFT on the rows and columns.
    """

    temp = np.array([ifft(row) for row in A])
    temp = np.array([ifft(col) for col in temp.T]).T

    return temp

def threshold_coeffs(A, keep_fraction):
    """
    Zeros out all but the largest-magnitude
    keep_fraction of coefficients.
    """

    flat = np.abs(A).ravel()
    n = flat.size
    k = max(int(np.floor(keep_fraction * n)), 1)
    thresh = np.partition(flat, -k)[-k]

    return A * (np.abs(A) >= thresh)

def compress_image(input_path, output_path, keep_fraction=0.1):
    """
    Reads an image, compresses it via 2D FFT thresholding,
    and saves the result.
    """

    # Load and convert to grayscale
    img = Image.open(input_path).convert('L')

    # Auto-resize to nearest lower powers of two
    orig_w, orig_h = img.size
    new_w = 2**int(math.floor(math.log2(orig_w)))
    new_h = 2**int(math.floor(math.log2(orig_h)))

    if (new_w, new_h) != (orig_w, orig_h):
        print(f"Resizing from {orig_w}×{orig_h} → {new_w}×{new_h}")
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    # Make array and proceed
    arr = np.array(img, dtype=float)
    h, w = arr.shape

    if not (h & (h - 1) == 0 and w & (w - 1) == 0):
        raise ValueError("Image dimensions must be powers of 2 for this FFT implementation.")

    # Forward FFT
    F = fft2d(arr)

    # Threshold coefficients
    F_thresh = threshold_coeffs(F, keep_fraction)

    # Inverse FFT
    arr_rec = ifft2d(F_thresh).real

    # Clip and convert back to uint8 image
    arr_rec = np.clip(arr_rec, 0, 255).astype(np.uint8)
    comp_img = Image.fromarray(arr_rec)
    comp_img.save(output_path)

    # Report compression ratio
    total_coeffs = F.size
    kept = np.count_nonzero(F_thresh)
    print(f"Kept {kept}/{total_coeffs} coefficients ({100 * kept / total_coeffs:.2f}% of original).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compress a grayscale image using 2D FFT thresholding."
    )
    parser.add_argument('input', help="Path to the input image (must be power-of-2 dimensions).")
    parser.add_argument('output', help="Path to save the compressed image.")
    parser.add_argument('--keep', type=float, default=0.1,
                        help="Fraction of FFT coefficients to keep (between 0 and 1).")
    args = parser.parse_args()
    compress_image(args.input, args.output, keep_fraction=args.keep)
