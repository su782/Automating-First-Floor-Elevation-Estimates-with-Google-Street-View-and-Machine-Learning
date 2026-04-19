import subprocess
import pandas as pd
import sys
import os


OUTPUT_ROOT = "outputs"


def run_pipeline_for_tract(tract_id):

    tract_root = os.path.join(OUTPUT_ROOT, tract_id)
    os.makedirs(tract_root, exist_ok=True)

    print("\n==============================")
    print(f"Running tract: {tract_id}")
    print("Output folder:", tract_root)
    print("==============================\n")

    env = os.environ.copy()
    env["PIPELINE_ROOT"] = tract_root
    env["TRACT_ID"] = tract_id

    subprocess.run(["python", "retrieve_addresses.py"], env=env)
    subprocess.run(["python", "front_view.py"], env=env)
    subprocess.run(["python", "select_best_views.py"], env=env)
    subprocess.run(["python", "refine_top1_with_sam.py"], env=env)
    subprocess.run(["python", "download_depthmaps_top1.py"], env=env)
    subprocess.run(["python", "estimate_ffe_rectified_from_sam.py"], env=env)

    print(f"\nFinished tract: {tract_id}\n")


def run_pipeline_multiple(tract_list):
    for tract_id in tract_list:
        run_pipeline_for_tract(str(tract_id))


def run_pipeline_from_csv(csv_path):
    df = pd.read_csv(csv_path)
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