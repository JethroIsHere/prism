"""Sanity check for the hand-written HOG implementation.

Runs BEFORE the full CIFAR-10 training, as required by the project brief:
the core classical algorithm must be tested on a small sanity check
(synthetic images here) before it touches the full test set.

Checks:
  1. Descriptor has the expected length (324 for 32x32, 8px cells, 9 bins).
  2. A blank image produces an all-zero descriptor.
  3. A vertical edge puts most gradient energy in bin 0 (0-20 deg).
  4. A horizontal edge puts most gradient energy in bin 4 (80-100 deg).
  5. Descriptors are finite and bounded (L2-Hys keeps every value <= 1).
  6. Determinism: same image -> identical descriptor.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hog_svm import hog_features, hog_dimension

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


def make_images():
    imgs = {}
    blank = np.zeros((32, 32, 3))
    vert = np.zeros((32, 32, 3)); vert[:, 16:, :] = 1.0     # left dark, right bright
    horiz = np.zeros((32, 32, 3)); horiz[16:, :, :] = 1.0   # top dark, bottom bright
    diag = np.zeros((32, 32, 3))
    yy, xx = np.mgrid[0:32, 0:32]
    diag[yy > xx, :] = 1.0
    # Anti-diagonal edge (top-right to bottom-left): its gradient is
    # perpendicular to it, i.e. 45 deg -> bins 2-3.
    adiag = np.zeros((32, 32, 3))
    adiag[yy + xx > 31, :] = 1.0
    imgs["blank"] = blank
    imgs["vertical_edge"] = vert
    imgs["horizontal_edge"] = horiz
    imgs["diagonal_edge"] = diag
    imgs["anti_diagonal_edge"] = adiag
    return imgs


def main():
    report = []
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        line = f"[{status}] {name}" + (f" -- {detail}" if detail else "")
        print(line, flush=True)
        report.append(line)

    imgs = make_images()
    names = list(imgs.keys())
    X = np.stack([imgs[k] for k in names])

    t0 = time.perf_counter()
    D = hog_features(X)
    dt = time.perf_counter() - t0

    check("descriptor length is 324", D.shape == (5, 324),
          f"got {D.shape}, expected (5, 324); hog_dimension()={hog_dimension()}")
    check("all descriptors finite", np.isfinite(D).all())
    check("L2-Hys bounds respected (|v| <= 1)", np.abs(D).max() <= 1.0 + 1e-9,
          f"max |v| = {np.abs(D).max():.4f}")
    check("blank image -> all-zero descriptor",
          np.abs(D[names.index("blank")]).max() == 0.0)

    # Per-cell histograms for the edge images: reshape (cells_y, cells_x, bins).
    def cell_hist(name):
        # Recompute per-block? Simpler: look at raw cells via a second call path.
        # Instead, verify via the dominant orientation of the whole descriptor's
        # energy per bin across the 9 blocks: bin energy summed over blocks.
        d = D[names.index(name)].reshape(9, 4, 9)  # 9 blocks, 2x2 cells, 9 bins
        return d.sum(axis=(0, 1))                  # total energy per orientation bin

    e_vert = cell_hist("vertical_edge")
    e_horiz = cell_hist("horizontal_edge")
    e_diag = cell_hist("diagonal_edge")
    e_adiag = cell_hist("anti_diagonal_edge")
    check("vertical edge -> bin 0 dominant (0-20 deg)",
          int(np.argmax(e_vert)) == 0,
          f"bin energies argmax={int(np.argmax(e_vert))}")
    check("horizontal edge -> bin 4 dominant (80-100 deg)",
          int(np.argmax(e_horiz)) == 4,
          f"bin energies argmax={int(np.argmax(e_horiz))}")
    # NOTE (corrected 2026-10-30): gradients are PERPENDICULAR to the edge.
    # The y=x diagonal has gradient orientation 135 deg -> bins 6-7.
    check("diagonal edge -> bins 6-7 dominant (gradients at 135 deg, "
          "perpendicular to the edge)",
          int(np.argmax(e_diag)) in (6, 7),
          f"bin energies argmax={int(np.argmax(e_diag))}")
    check("anti-diagonal edge -> bins 2-3 dominant (gradients at 45 deg)",
          int(np.argmax(e_adiag)) in (2, 3),
          f"bin energies argmax={int(np.argmax(e_adiag))}")

    D2 = hog_features(X)
    check("deterministic output", np.array_equal(D, D2))
    check("runs fast enough for 60k images",
          dt / 5 * 60000 < 1800,
          f"{dt / 5 * 1000:.1f} ms/image -> ~{dt / 5 * 60000 / 60:.0f} min for 60k")

    report.append(f"\nSanity check ran in {dt:.2f}s for 5 images.")
    with open(os.path.join(RESULTS, "sanity_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")

    print("\nOVERALL:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
