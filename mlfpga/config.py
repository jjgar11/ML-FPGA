import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MODELS_ROOT = os.path.join(DATA_ROOT, "models")
HLS4ML_ROOT = os.path.join(DATA_ROOT, "hls4ml_prj")
FPGA_FILES_ROOT = os.path.join(PROJECT_ROOT, "FPGA-files")

# data/ is gitignored (raw datasets, trained weights — regenerated or copied per
# machine). assets/ is tracked: small static reference files scripts need to run
# anywhere the repo is cloned (class name tables, fixed test sets).
ASSETS_ROOT = os.path.join(PROJECT_ROOT, "assets")
