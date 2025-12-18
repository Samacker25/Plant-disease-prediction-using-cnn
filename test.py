from huggingface_hub import snapshot_download
import os

repo_dir = snapshot_download(
    repo_id="Samacker25/plant-disease-prediction",
    repo_type="dataset"
)

print("Downloaded to:", repo_dir)

for root, dirs, files in os.walk(repo_dir):
    level = root.replace(repo_dir, "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        print(f"{indent}  {f}")
