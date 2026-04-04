import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: huggingface_hub not found. Run: .venv/bin/pip install huggingface_hub")
    sys.exit(1)

def deploy():
    print("🚀 Cattle Breed Classifier Hugging Face Deployer")
    print("-" * 50)
    
    # 1. Get credentials securely
    username = input("Enter your Hugging Face Username: ").strip()
    space_name = input("Enter your desired Space Name (e.g. cattle-classifier): ").strip()
    token = input("Enter your Hugging Face Write Token: ").strip()
    
    if not username or not space_name or not token:
        print("Error: All fields are required.")
        sys.exit(1)
        
    repo_id = f"{username}/{space_name}"
    api = HfApi(token=token)
    
    print(f"\n[1/3] Creating Space '{repo_id}' on Hugging Face...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            token=token,
            exist_ok=True
        )
        print("✅ Space created/verified successfully!")
    except Exception as e:
        print(f"❌ Failed to create space: {e}")
        sys.exit(1)
        
    print("\n[2/3] Uploading Application & Massive ViT Model...")
    print("⏳ This may take several minutes depending on your internet upload speed (Model is ~460MB). Please wait...")
    
    # Files to ignore (don't upload heavy training folders or virtual environments)
    ignore_patterns = [
        ".venv/*",
        ".git/*",
        "__pycache__/*",
        "*.pyc",
        "Cattle_Resized/*",
        "ml/artifacts/checkpoints/cnn_best.pth",
        "ml/artifacts/checkpoints/mlp_best.pth",
        "ml/artifacts/checkpoints/resnet_best.pth",
        # Ignore models dir except vit_best.pth and classes.txt
        "frontend/node_modules/*"
    ]
    
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="Initial Deployment: Best ViT Model"
        )
        print("✅ Upload complete!")
    except Exception as e:
        print(f"❌ Failed to upload files: {e}")
        sys.exit(1)
        
    print(f"\n[3/3] Deployment Successful! 🎉")
    print(f"Your application is now building in the cloud.")
    print(f"View it live here: https://huggingface.co/spaces/{username}/{space_name}")

if __name__ == "__main__":
    deploy()
