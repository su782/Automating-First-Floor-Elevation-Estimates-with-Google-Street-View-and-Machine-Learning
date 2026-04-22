import subprocess
import pandas as pd
import sys
import os

OUTPUT_ROOT = "outputs"

# =========================================================
# 在这里统一设置整个 pipeline 需要的配置
# =========================================================
GOOGLE_API_KEY = "AIzaSyBtuibB4nNTRnyXDnQgDq1nZpYffiOaAa8"
DETECTRON_MODEL_PATH = r"C:\capstone\ffe_pipeline\output_cvat200_from_old4class\model_final.pth"
SAM_CHECKPOINT_PATH = r"C:\capstone\ffe_pipeline\sam_vit_h_4b8939.pth"

# 关键：detectron2 源码目录
DETECTRON2_SRC = r"C:\capstone\ffe_pipeline\detectron2-main"

# 如果你想强制用 cpu，也可以在这里传
# 例如 "cpu" 或 "cuda"
SAM_DEVICE = "cuda"

SCRIPT_RETRIEVE = "nsi_retriever.py"
SCRIPT_GSV = "gsv_download_walk15.py"
SCRIPT_SELECT = "front_view.py"
SCRIPT_SAM = "refine_top1_with_sam.py"
SCRIPT_DEPTH = "download_depthmaps_top1.py"
SCRIPT_FFE = "estimate_ffe_rectified_from_sam.py"


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

    if not os.path.exists(DETECTRON_MODEL_PATH):
        raise FileNotFoundError(f"Detectron model not found: {DETECTRON_MODEL_PATH}")

    if not os.path.exists(SAM_CHECKPOINT_PATH):
        raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CHECKPOINT_PATH}")

    if not os.path.isdir(DETECTRON2_SRC):
        raise FileNotFoundError(f"Detectron2 source dir not found: {DETECTRON2_SRC}")


def build_pipeline_env(tract_root, tract_id):
    env = os.environ.copy()

    # pipeline 基本信息
    env["PIPELINE_ROOT"] = tract_root
    env["TRACT_ID"] = str(tract_id)

    # 统一传给所有模块
    env["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    env["DETECTRON_MODEL_PATH"] = DETECTRON_MODEL_PATH
    env["SAM_CHECKPOINT_PATH"] = SAM_CHECKPOINT_PATH
    env["SAM_DEVICE"] = SAM_DEVICE
    env["DETECTRON2_SRC"] = DETECTRON2_SRC

    # 保险：把 detectron2 源码目录也塞进 PYTHONPATH
    env["PYTHONPATH"] = DETECTRON2_SRC + os.pathsep + env.get("PYTHONPATH", "")

    return env


def stop_if_no_addresses(tract_root, tract_id):
    addresses_csv = os.path.join(tract_root, "addresses.csv")
    if not os.path.exists(addresses_csv):
        raise RuntimeError("addresses.csv not created by nsi_retriever.py")

    df_addr = pd.read_csv(addresses_csv)
    print(f"[INFO] addresses.csv rows = {len(df_addr)}")

    if len(df_addr) == 0:
        print(f"[STOP] No addresses found for tract {tract_id}. Skip remaining steps.")
        return True

    return False


def check_gsv_images_exist(tract_root):
    gsv_root = os.path.join(tract_root, "gsv_images")
    if not os.path.isdir(gsv_root):
        raise RuntimeError(f"gsv_images folder not found: {gsv_root}")

    subdirs = [f.path for f in os.scandir(gsv_root) if f.is_dir()]
    print(f"[INFO] gsv address folders = {len(subdirs)}")
    return gsv_root


def run_pipeline_for_tract(tract_id):
    tract_id = str(tract_id)
    tract_root = os.path.join(OUTPUT_ROOT, tract_id)
    os.makedirs(tract_root, exist_ok=True)

    print("\n==============================")
    print(f"Running tract: {tract_id}")
    print(f"Output folder: {tract_root}")
    print("==============================\n")

    check_required_files()
    env = build_pipeline_env(tract_root, tract_id)

    run_step(SCRIPT_RETRIEVE, env)

    if stop_if_no_addresses(tract_root, tract_id):
        return

    run_step(SCRIPT_GSV, env)
    check_gsv_images_exist(tract_root)

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