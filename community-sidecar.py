#!/usr/bin/env python3
"""community-sidecar.py

Optional Python timing sidecar for the community differentiation benchmark.
Self-times the per-stage cost of a single (backend, problem, n) cell and emits
one JSON object on stdout for the R harness to fold into the run-log:

    {"system": "PyTorch", "version": "2.4.0",
     "stages": [{"stage": "import",  "median_ms": 950.0, "reps": 1},
                {"stage": "trace",   "median_ms": 12.0,  "reps": 1},
                {"stage": "eval",    "median_ms": 81.7,  "iqr_ms": .., "cv_pct": .., "reps": 20}]}

Startup stages (import, trace/jit_compile) are measured, not discarded — the
whole point of the community benchmark is the cold-start story. float64 is
pinned so the comparison matches R's double precision.

If the backend is not installed, prints {"error": "<backend> not installed"}
and exits 0 so the R harness degrades gracefully.

Usage:
    python3 community-sidecar.py --backend {torch|jax} --problem sum_v2 \
            --n 1000000 [--threads 1] [--reps 20]
"""
import argparse
import json
import statistics
import sys
import time

PROBLEMS = ("sum_v2", "sum_v3", "sum_sin_v")


def _summary(times_ms):
    """median / IQR / CV over a list of per-iteration times in ms."""
    s = sorted(times_ms)
    n = len(s)
    med = statistics.median(s)
    if n >= 4:
        q1 = s[n // 4]
        q3 = s[(3 * n) // 4]
        iqr = q3 - q1
    else:
        iqr = None
    mean = statistics.fmean(s) if s else float("nan")
    sd = statistics.pstdev(s) if n > 1 else 0.0
    cv = (100.0 * sd / mean) if mean else None
    return med, iqr, cv


def run_torch(problem, n, threads, reps):
    t0 = time.perf_counter()
    import torch  # noqa: E402
    import_ms = (time.perf_counter() - t0) * 1000.0
    if threads:
        torch.set_num_threads(int(threads))
    torch.set_default_dtype(torch.float64)
    from torch.func import grad

    funcs = {
        "sum_v2":    lambda v: (v ** 2).sum(),
        "sum_v3":    lambda v: (v ** 3).sum(),
        "sum_sin_v": lambda v: torch.sin(v).sum(),
    }
    f = funcs[problem]
    v = torch.rand(n, dtype=torch.float64)
    g = grad(f)

    t0 = time.perf_counter()
    g(v)
    trace_ms = (time.perf_counter() - t0) * 1000.0

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        g(v)
        times.append((time.perf_counter() - t0) * 1000.0)
    med, iqr, cv = _summary(times)
    return {
        "system": "PyTorch",
        "version": torch.__version__,
        "stages": [
            {"stage": "import", "median_ms": import_ms, "reps": 1},
            {"stage": "trace", "median_ms": trace_ms, "reps": 1},
            {"stage": "eval", "median_ms": med, "iqr_ms": iqr, "cv_pct": cv, "reps": reps},
        ],
    }


def run_jax(problem, n, threads, reps):
    import os
    if threads:
        os.environ.setdefault("OMP_NUM_THREADS", str(int(threads)))
    t0 = time.perf_counter()
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402
    import_ms = (time.perf_counter() - t0) * 1000.0
    jax.config.update("jax_enable_x64", True)

    funcs = {
        "sum_v2":    lambda v: jnp.sum(v ** 2),
        "sum_v3":    lambda v: jnp.sum(v ** 3),
        "sum_sin_v": lambda v: jnp.sum(jnp.sin(v)),
    }
    f = funcs[problem]
    v = jnp.asarray(np.random.rand(n))
    g = jax.jit(jax.grad(f))

    t0 = time.perf_counter()
    g(v).block_until_ready()
    jit_ms = (time.perf_counter() - t0) * 1000.0

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        g(v).block_until_ready()
        times.append((time.perf_counter() - t0) * 1000.0)
    med, iqr, cv = _summary(times)
    return {
        "system": "JAX",
        "version": jax.__version__,
        "stages": [
            {"stage": "import", "median_ms": import_ms, "reps": 1},
            {"stage": "jit_compile", "median_ms": jit_ms, "reps": 1},
            {"stage": "eval", "median_ms": med, "iqr_ms": iqr, "cv_pct": cv, "reps": reps},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=("torch", "jax"))
    ap.add_argument("--problem", required=True, choices=PROBLEMS)
    ap.add_argument("--n", required=True, type=int)
    ap.add_argument("--threads", default=0, type=int)
    ap.add_argument("--reps", default=20, type=int)
    args = ap.parse_args()
    try:
        if args.backend == "torch":
            result = run_torch(args.problem, args.n, args.threads, args.reps)
        else:
            result = run_jax(args.problem, args.n, args.threads, args.reps)
    except ImportError:
        print(json.dumps({"error": f"{args.backend} not installed"}))
        return 0
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
