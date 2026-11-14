"""Egg roulette: learning grip force from human touch demonstrations, in MuJoCo."""
import os

# experiments run one process per core: keep BLAS and torch from spawning threads inside each
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
