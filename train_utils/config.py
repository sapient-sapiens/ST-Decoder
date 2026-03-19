"""Hyperparameters for the paper experiments (spatio-temporal local model + baselines)."""

from pathlib import Path

# Repository root (parent of the `train_utils` package).
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Spatio-temporal local (main model): masked spatial + sliding-window temporal ---
ST_LOCAL_D_MODEL = 240
ST_LOCAL_N_LAYERS = 4
ST_LOCAL_S_NHEAD = 8
ST_LOCAL_T_NHEAD = 4
ST_LOCAL_FFN_DIM = 1024
ST_LOCAL_SPATIAL_RADIUS = 4
ST_TEMPORAL_WINDOW = 3

# --- Vanilla transformer (baseline) ---
VANILLA_D_MODEL = 272
VANILLA_N_LAYERS = 5
VANILLA_N_HEAD = 8
VANILLA_FFN_DIM = 1152

# --- GNN (baseline) ---
GNN_HIDDEN_GCN = [640, 640, 640, 640, 640, 640]
GNN_HIDDEN_MLP = [512, 256]
GNN_DROPOUT = 0.1

# --- Training defaults ---
DROPOUT = 0.1
LR = 1e-4
WEIGHT_DECAY = 0.0
BATCH_SIZE = 512
VAL_RATIO = 0.1
NUM_WORKERS = 0

PS = [0.002, 0.005, 0.008]
PAIRS = [(3, 3), (3, 5), (3, 7), (5, 3), (5, 5), (5, 7), (7, 3), (7, 5), (7, 7)]
