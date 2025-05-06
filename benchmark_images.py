import time, math, argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from compress import fft2d, threshold_coeffs, ifft2d

def compress_array(arr, keep_fraction=0.5):
    """
    Main compressor working on 2D array
    """

    F = fft2d(arr)
    Fth = threshold_coeffs(F, keep_fraction)

    return ifft2d(Fth).real

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FFT compressor on real images at multiple sizes."
    )
    parser.add_argument('image',
        help="Path to your original image (e.g. Lena.jpeg or Mandrill.jpg).")
    parser.add_argument('--keep', type=float, default=0.5,
        help="Fraction of coefficients to keep for the benchmark.")
    parser.add_argument('--sizes', nargs='+', type=int,
        default=[64, 128, 256, 512],
        help="List of powers-of-two sizes to test (rows=cols).")
    args = parser.parse_args()

    """
    Load and grayscale once
    """
    orig = Image.open(args.image).convert('L')

    results = []

    for n in args.sizes:
        """
        Resize to n x n
        """
        im_n = orig.resize((n, n), resample=Image.LANCZOS)
        arr  = np.array(im_n, float)
        t0 = time.time()
        _  = compress_array(arr, args.keep)
        elapsed = time.time() - t0
        M = n * n
        results.append({
            'n': n,
            'M': M,
            'time_s': elapsed,
            'M_log2_M': M * math.log2(M)
        })

        print(f"Size {n}×{n}: time = {elapsed:.4f}s")

    """
    Buld Data-Frame summary and CSV
    """
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv('benchmark_images.csv', index=False)
        print("Wrote benchmark_images.csv")
    except ImportError:
        pass

    """
    Time vs M
    """
    plt.figure()
    plt.plot([r['M'] for r in results],
             [r['time_s'] for r in results], 'o-')
    plt.xlabel('Pixels $M$')
    plt.ylabel('Time (s)')
    plt.title(f'FFT Compressor Timing (keep={args.keep})')
    plt.savefig('time_vs_M.png')
    plt.close()
    print("Saved time_vs_M.png")

    """
    Time vs M log2 M
    """
    plt.figure()
    plt.plot([r['M_log2_M'] for r in results],
             [r['time_s']    for r in results], 'o-')
    plt.xlabel('$M \\log_2 M$')
    plt.ylabel('Time (s)')
    plt.title(f'FFT Compressor Timing (keep={args.keep})')
    plt.savefig('time_vs_MlogM.png')
    plt.close()
    print("Saved time_vs_MlogM.png")

if __name__ == '__main__':
    main()
    
