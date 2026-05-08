"""
run_ae_pipeline.py — End-to-end orchestrator for the encoder-decoder pipeline.

Stages (all sequential — AE fine-tunes the full backbone so parallel runs
would compete for the same GPU memory):
  1. Train AE dim=512   (~60-90 min on GPU)
  2. Train AE dim=256   (~60-90 min on GPU)
  3. Extract embeddings — 512-d, 256-d, PCA   (~5 min)
  4. Train MLP on 512-d, 256-d, PCA in parallel   (~2-5 min each)
  5. Evaluate all three MLPs + write ae_comparison.md

Usage:
    python run_ae_pipeline.py
    python run_ae_pipeline.py --smoke-test          # 1-epoch AE sanity check
    python run_ae_pipeline.py --skip-ae             # skip AE training
    python run_ae_pipeline.py --skip-ae --skip-emb  # only MLP train + eval
"""

import argparse
import subprocess
import sys
import time
import threading
from pathlib import Path


def run(cmd: list, label: str, log_path: Path = None):
    """Run a subprocess, optionally capturing output to a log file."""
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    t0 = time.time()
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=lf)
        # Stream last few lines to console so user sees progress
        lines = log_path.read_text(errors="replace").splitlines()
        for line in lines[-5:]:
            print(f"  {line}")
    else:
        proc = subprocess.run(cmd)
    elapsed = time.time() - t0
    status  = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"[{status}] {label}  ({elapsed/60:.1f} min)")
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {label}")
    return proc


def run_parallel(cmds_labels: list[tuple], log_dir: Path):
    """Run multiple subprocesses in parallel; wait for all."""
    procs = []
    for cmd, label in cmds_labels:
        log = log_dir / f"{label.replace(' ', '_')}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        lf = open(log, "w")
        p  = subprocess.Popen(cmd, stdout=lf, stderr=lf)
        procs.append((p, lf, label, log))
        print(f"[launched] {label}  (pid={p.pid})")

    failed = []
    while procs:
        for item in list(procs):
            p, lf, label, log = item
            if p.poll() is not None:
                lf.close()
                status = "done" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
                print(f"[{status}] {label}")
                if p.returncode != 0:
                    failed.append(label)
                procs.remove(item)
        if procs:
            time.sleep(5)

    if failed:
        raise RuntimeError(f"Steps failed: {failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test",  action="store_true")
    parser.add_argument("--skip-ae",     action="store_true")
    parser.add_argument("--skip-emb",    action="store_true")
    args = parser.parse_args()

    py = sys.executable

    # ── Stage 1 & 2: Train AE (sequential — GPU memory) ──
    if not args.skip_ae:
        for dim in [512, 256]:
            cmd = [py, "train_ae.py", "--dim", str(dim)]
            if args.smoke_test:
                cmd.append("--smoke-test")
            run(cmd,
                label=f"AE training  dim={dim}",
                log_path=Path(f"reports/exp_ae_{dim}/run_train.log"))

    # ── Stage 3: Extract embeddings ──
    if not args.skip_emb:
        run([py, "extract_embeddings.py"],
            label="Extract embeddings (512, 256, PCA)",
            log_path=Path("reports/extract_embeddings.log"))

    # ── Stage 4: Train MLPs in parallel (CPU-bound, no GPU contention) ──
    print("\n=== Training MLPs in parallel ===")
    run_parallel(
        [([py, "train_mlp.py", "--emb", tag], f"MLP {tag}")
         for tag in ["512", "256", "pca"]],
        log_dir=Path("reports"),
    )

    # ── Stage 5: Evaluate all + comparison report ──
    run([py, "evaluate_mlp.py", "--all"],
        label="Evaluate all MLPs + write ae_comparison.md",
        log_path=Path("reports/evaluate_mlp.log"))

    print("\nPipeline complete. See reports/ae_comparison.md for results.")


if __name__ == "__main__":
    main()
