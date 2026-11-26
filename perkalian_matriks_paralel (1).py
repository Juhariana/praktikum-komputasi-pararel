"""
Praktikum Komputasi Paralel: Perkalian Matriks
Nama : Juhariana
NIM  : D0224336
Kelas: 2024D

Program membandingkan perkalian matriks secara sekuensial
dan paralel menggunakan multiprocessing.
"""

import argparse
import multiprocessing as mp
import random
import time


def multiply_rows(task):
    """Menghitung beberapa baris hasil perkalian matriks."""
    a, b_transposed, start_row, end_row = task
    result_part = []

    for i in range(start_row, end_row):
        row = []
        for col in b_transposed:
            row.append(
                sum(a[i][k] * col[k] for k in range(len(col)))
            )
        result_part.append(row)

    return start_row, result_part


def sequential_multiply(a, b):
    """Perkalian matriks secara sekuensial."""
    b_transposed = list(zip(*b))
    result = []

    for row in a:
        result.append([
            sum(x * y for x, y in zip(row, col))
            for col in b_transposed
        ])

    return result


def parallel_multiply(a, b, workers):
    """Perkalian matriks menggunakan multiprocessing."""
    n = len(a)
    b_transposed = [list(col) for col in zip(*b)]

    chunk_size = (n + workers - 1) // workers
    tasks = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        tasks.append((a, b_transposed, start, end))

    result = [None] * n

    with mp.Pool(processes=workers) as pool:
        for start_row, part in pool.map(multiply_rows, tasks):
            result[start_row:start_row + len(part)] = part

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Perbandingan perkalian matriks sekuensial dan paralel."
    )

    parser.add_argument(
        "--size",
        type=int,
        default=140,
        help="Ukuran matriks (default: 140)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Jumlah proses paralel (default: 4)"
    )

    # Mengabaikan argumen tambahan milik Google Colab/Jupyter,
    # tetapi tetap menerima --size dan --workers saat dijalankan sebagai script.
    args, _ = parser.parse_known_args()

    if args.size < 1 or args.workers < 1:
        raise SystemExit("Ukuran dan jumlah proses harus lebih dari 0.")

    random.seed(42)

    a = [
        [random.randint(1, 9) for _ in range(args.size)]
        for _ in range(args.size)
    ]

    b = [
        [random.randint(1, 9) for _ in range(args.size)]
        for _ in range(args.size)
    ]

    start = time.perf_counter()
    result_seq = sequential_multiply(a, b)
    sequential_time = time.perf_counter() - start

    start = time.perf_counter()
    result_par = parallel_multiply(a, b, args.workers)
    parallel_time = time.perf_counter() - start

    correct = result_seq == result_par
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0

    print("=== HASIL PRAKTIKUM KOMPUTASI PARALEL ===")
    print()
    print(f"Ukuran matriks        : {args.size} x {args.size}")
    print(f"Jumlah proses         : {args.workers}")
    print(f"Waktu sekuensial      : {sequential_time:.6f} detik")
    print(f"Waktu paralel         : {parallel_time:.6f} detik")
    print(f"Speedup               : {speedup:.3f} kali")
    print()
    print(
        "Pemeriksaan hasil    :",
        "SAMA (BERHASIL)" if correct else "TIDAK SAMA (GAGAL)"
    )

    if not correct:
        raise SystemExit("Hasil perhitungan tidak sama.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
