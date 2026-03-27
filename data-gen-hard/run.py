"""
run.py — Generate the full dataset.

Usage:
    python run.py                          # full run, 2 workers
    python run.py --single                 # single-process debug
    python run.py --out my/output/dir
    python run.py --workers 4
"""
import sys
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.shard_writer import generate_dataset

if __name__ == "__main__":
    mp.freeze_support()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",     default="outputs/shards")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--single",  action="store_true")
    args = parser.parse_args()
    generate_dataset(output_dir=args.out, n_workers=1 if args.single else args.workers)
