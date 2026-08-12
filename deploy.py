"""
PlantSeg Deployment Script
==========================
Automated deployment to Hugging Face Spaces + GitHub.

Usage:
    python deploy.py \\
        --hf-token YOUR_HF_TOKEN \\
        --hf-space username/plantseg-demo \\
        --hf-model-repo username/plantseg-models \\
        --github-token YOUR_GITHUB_TOKEN \\
        --github-repo username/PlantSeg-Decision-Support \\
        --cls-model path/to/ConvNeXtV2Tiny_best.pt \\
        --seg-model path/to/DeepLabV3Plus_efficientnet-b3.pt
"""
import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path


def run(cmd, **kwargs):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    return result


def main():
    parser = argparse.ArgumentParser(description="Deploy PlantSeg to HF Spaces + GitHub")
    parser.add_argument("--hf-token", required=True, help="Hugging Face write token")
    parser.add_argument("--hf-space", required=True, help="HF Space name (e.g., user/plantseg-demo)")
    parser.add_argument("--hf-model-repo", required=True, help="HF model repo (e.g., user/plantseg-models)")
    parser.add_argument("--github-token", required=True, help="GitHub personal access token")
    parser.add_argument("--github-repo", required=True, help="GitHub repo (e.g., user/PlantSeg)")
    parser.add_argument("--cls-model", required=True, help="Path to ConvNeXtV2Tiny_best.pt")
    parser.add_argument("--seg-model", required=True, help="Path to DeepLabV3Plus_efficientnet-b3.pt")
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    
    # ---- Step 1: Upload models to HF Hub ----
    print("\n" + "="*60)
    print("STEP 1: Uploading models to Hugging Face Hub")
    print("="*60)
    
    try:
        from huggingface_hub import HfApi
    except ImportError:
        run(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import HfApi
    
    api = HfApi(token=args.hf_token)
    
    # Create model repo if it doesn't exist
    try:
        api.create_repo(repo_id=args.hf_model_repo, repo_type="model", exist_ok=True)
        print(f"Model repo ready: {args.hf_model_repo}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload model files
    for model_path, model_name in [
        (args.cls_model, "ConvNeXtV2Tiny_best.pt"),
        (args.seg_model, "DeepLabV3Plus_efficientnet-b3.pt")
    ]:
        if not Path(model_path).exists():
            print(f"ERROR: Model file not found: {model_path}")
            sys.exit(1)
        print(f"Uploading {model_name}...")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_name,
            repo_id=args.hf_model_repo,
            repo_type="model",
        )
        print(f"  ✓ {model_name} uploaded")
    
    # ---- Step 2: Create and deploy HF Space ----
    print("\n" + "="*60)
    print("STEP 2: Deploying to Hugging Face Spaces")
    print("="*60)
    
    try:
        api.create_repo(repo_id=args.hf_space, repo_type="space", space_sdk="gradio", exist_ok=True)
        print(f"Space ready: {args.hf_space}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Set the HF_MODEL_REPO secret on the space
    try:
        api.add_space_secret(repo_id=args.hf_space, key="HF_MODEL_REPO", value=args.hf_model_repo)
        print(f"  ✓ HF_MODEL_REPO secret set to {args.hf_model_repo}")
    except Exception as e:
        print(f"  Note (secret): {e}")
    
    # Upload app files to the Space
    files_to_upload = ["app.py", "requirements.txt", "class_names.json", "README.md"]
    for fname in files_to_upload:
        fpath = project_dir / fname
        if fpath.exists():
            print(f"Uploading {fname} to Space...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=args.hf_space,
                repo_type="space",
            )
            print(f"  ✓ {fname} uploaded")
    
    # ---- Step 3: Push to GitHub ----
    print("\n" + "="*60)
    print("STEP 3: Pushing to GitHub")
    print("="*60)
    
    github_user = args.github_repo.split("/")[0]
    github_repo_name = args.github_repo.split("/")[1]
    remote_url = f"https://{args.github_token}@github.com/{args.github_repo}.git"
    
    os.chdir(project_dir)
    
    # Initialize git if needed
    if not (project_dir / ".git").exists():
        run("git init")
        run("git branch -M main")
    
    # Create the repo on GitHub via API
    import urllib.request
    import json
    
    req = urllib.request.Request(
        "https://api.github.com/user/repos",
        data=json.dumps({"name": github_repo_name, "private": False, "description": "PlantSeg Faithfulness-Gated Decision Support - Plant Disease Diagnosis"}).encode(),
        headers={"Authorization": f"token {args.github_token}", "Content-Type": "application/json"},
        method="POST"
    )
    try:
        urllib.request.urlopen(req)
        print(f"GitHub repo created: {args.github_repo}")
    except Exception as e:
        print(f"Note (repo may already exist): {e}")
    
    # Add remote and push
    run(f'git remote remove origin 2>nul & git remote add origin "{remote_url}"')
    run("git add -A")
    run('git commit -m "Initial deployment: PlantSeg Decision Support"')
    run("git push -u origin main --force")
    
    # ---- Done ----
    print("\n" + "="*60)
    print("DEPLOYMENT COMPLETE!")
    print("="*60)
    print(f"\n  🌿 Hugging Face Space: https://huggingface.co/spaces/{args.hf_space}")
    print(f"  📦 Model Repository:   https://huggingface.co/{args.hf_model_repo}")
    print(f"  💻 GitHub Repository:  https://github.com/{args.github_repo}")
    print(f"\nYour Space will build automatically. Check the Spaces page for build status.")


if __name__ == "__main__":
    main()
