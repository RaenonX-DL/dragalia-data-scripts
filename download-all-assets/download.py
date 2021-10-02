from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
from typing import Generator
from urllib.request import urlopen

BASE = "https://dragalialost.akamaized.net/dl"


def get_asset_url(asset_hash: str) -> str:
    return f"{BASE}/assetbundles/Android/{asset_hash[:2]}/{asset_hash}"
    
    
def get_file_path(download_dir: str, asset_hash: str) -> str:
    return f"{download_dir}/{asset_hash[:2]}/{asset_hash}"


def download_asset(file_path: str, url: str):
    if os.path.exists(file_path):
        return
        
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if needed

    with open(file_path, "wb") as f:
        f.write(urlopen(url).read())
        
        
def get_asset_entries(download_dir: str, manifest_path: str) -> Generator[str, None, None]:
    with open(manifest_path) as f:
        json_dict = json.load(f)
        
        parents = (
            json_dict["categories"][1]["assets"],
            json_dict["rawAssets"]
        )
        
        for parent in parents:
            for asset in parent:
                asset_hash = asset["hash"]
                file_path = get_file_path(download_dir, asset_hash)
                yield (file_path, get_asset_url(asset_hash))


def main():
    manifest = sys.argv[1]
    download_dir = sys.argv[2]

    with ThreadPoolExecutor(max_workers=32) as executor:
        print("Queueing the tasks...")
        futures = [
            executor.submit(download_asset, *asset) 
            for asset in get_asset_entries(download_dir, manifest)
        ]
        print("Tasks queued.")

        for idx, future in enumerate(futures):
            future.result()
            print(f"{idx} / {len(futures)} ({idx / len(futures):.2%} completed)")


if __name__ == '__main__':
    main()
