import subprocess
import pandas as pd
import sys
import os
import urllib.request

# =========================================================
# 基础配置
# =========================================================

OUTPUT_ROOT = "outputs"

MODEL_URL = (
    "https://github.com/su782/"
    "Automating-First-Floor-Elevation-Estimates-with-Google-Street-View-and-Machine-Learning/"
    "releases/download/model/model_final.pth"
)

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

DETECTRON_MODEL_PATH = os.path.join(MODEL_DIR, "model_final.pth")
SAM_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "sam_vit_h_4b8939.pth")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
SAM_DEVICE = "cuda"  

SCRIPT_RETRIEVE = "nsi_retriever.py"
SCRIPT_GSV = "gsv_download_walk15.py"
SCRIPT_SELECT = "front_view.py"
SCRIPT_SAM = "refine_top1_with_sam.py"
SCRIPT_DEPTH = "download_depthmaps_top1.py"
SCRIPT_FFE = "estimate_ffe_rectified_from_sam.py"



def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        print(f"Downloading: {os.path.basename(save_path)}")
        urllib.request.urlretrieve(url, save_path)
        print("Download complete.")


def ensure_models():
    download_file(MODEL_URL, DETECTRON_MODEL_PATH)
    download_file(SAM_URL, SAM_CHECKPOINT_PATH)



def run_step(script_name, env):
    cmd = [sys.executable, script_name]
    print(f"\n[RUN ] {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed: {script_name} (return code={result.returncode})"
        )

    print(f"[DONE] {script_name}")


def check_required_files():
    required = [
        SCRIPT_RETRIEVE,
        SCRIPT_GSV,
        SCRIPT_SELECT,
        SCRIPT_SAM,
        SCRIPT_DEPTH,
        SCRIPT_FFE,
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing required scripts: {missing}")


def build_pipeline_env(tract_root, tract_id):
    env = os.environ.copy()

    env["PIPELINE_ROOT"] = tract_root
    env["TRACT_ID"] = str(tract_id)

    env["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    env["DETECTRON_MODEL_PATH"] = DETECTRON_MODEL_PATH
    env["SAM_CHECKPOINT_PATH"] = SAM_CHECKPOINT_PATH
    env["SAM_DEVICE"] = SAM_DEVICE

    return env


def stop_if_no_addresses(tract_root, tract_id):
    addresses_csv = os.path.join(tract_root, "addresses.csv")
    if not os.path.exists(addresses_csv):
        raise RuntimeError("addresses.csv not created by nsi_retriever.py")

    df_addr = pd.read_csv(addresses_csv)
    print(f"[INFO] addresses.csv rows = {len(df_addr)}")

    if len(df_addr) == 0:
        print(f"[STOP] No addresses found for tract {tract_id}.")
        return True

    return False


def run_pipeline_for_tract(tract_id):
    tract_id = str(tract_id)
    tract_root = os.path.join(OUTPUT_ROOT, tract_id)
    os.makedirs(tract_root, exist_ok=True)

    print("\n==============================")
    print(f"Running tract: {tract_id}")
    print(f"Output folder: {tract_root}")
    print("==============================\n")

    check_required_files()
    ensure_models()

    env = build_pipeline_env(tract_root, tract_id)

    run_step(SCRIPT_RETRIEVE, env)

    if stop_if_no_addresses(tract_root, tract_id):
        return

    run_step(SCRIPT_GSV, env)
    run_step(SCRIPT_SELECT, env)
    run_step(SCRIPT_SAM, env)
    run_step(SCRIPT_DEPTH, env)
    run_step(SCRIPT_FFE, env)

    print(f"\nFinished tract: {tract_id}\n")


def run_pipeline_multiple(tract_list):
    for tract_id in tract_list:
        run_pipeline_for_tract(tract_id)


def run_pipeline_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "tract_id" not in df.columns:
        raise ValueError("CSV must contain a column named 'tract_id'")
    tract_list = df["tract_id"].astype(str).tolist()
    run_pipeline_multiple(tract_list)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_pipeline.py 36061000100")
        print("  python run_pipeline.py 36061000100 36061000200")
        print("  python run_pipeline.py tract_list.csv")
        sys.exit(1)

    args = sys.argv[1:]

    if len(args) == 1 and args[0].endswith(".csv"):
        run_pipeline_from_csv(args[0])
    else:
        run_pipeline_multiple(args)