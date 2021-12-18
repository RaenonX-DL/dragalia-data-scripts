from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
from typing import Generator
from urllib.request import urlopen

UNIT_IDS = [
    # Flame (19)
    10150106,
    10150103,
    10150101,
    10250104,
    10250102,
    10250101,
    10350101,
    10450103,
    10450101,
    10550103,
    10550102,
    10650103,
    10650102,
    10750105,
    10750103,
    10750101,
    10850102,
    10850101,
    10950102,
    # Water (21)
    10150203,
    10150202,
    10150201,
    10250204,
    10250201,
    10350204,
    10350202,
    10450204,
    10450203,
    10450201,
    10550205,
    10550204,
    10550201,
    10650203,
    10650201,
    10750202,
    10750201,
    10850203,
    10850201,
    10950203,
    10950201,
    # Wind (25)
    10150305,
    10150302,
    10250304,
    10250303,
    10250302,
    10250301,
    10350302,
    10350301,
    10450304,
    10450301,
    10550304,
    10550302,
]

CHARA_DATA = "https://raw.githubusercontent.com/RaenonX-DL/dragalia-data-depot/main/assets/_gluonresources/resources" \
             "/master/CharaData.json"


def get_image_url(unit_icon_name: str) -> str:
    return f"https://raw.githubusercontent.com/RaenonX-DL/dragalia-data-depot/main/assets/_gluonresources/resources" \
           f"/images/icon/chara/l/{unit_icon_name}.png"


def download_asset(file_path: str, url: str):
    if os.path.exists(file_path):
        return

    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if needed

    with open(file_path, "wb") as f:
        f.write(urlopen(url).read())


def get_asset_entries(download_dir: str, manifest_path: str) -> Generator[str, None, None]:
    with open(manifest_path) as f:
        json_dict = json.load(f)

        d = {val["_Id"]: val for val in json_dict["dict"]["entriesValue"]}

        for unit_id in UNIT_IDS:
            entry = d[unit_id]
            icon_name = f"{entry['_BaseId']}_{entry['_VariationId']:02}_r{entry['_Rarity']:02}"

            file_path = os.path.join(download_dir, f"{icon_name}.png")
            yield file_path, get_image_url(icon_name)


def main():
    manifest = r"D:\UserData\Downloads\CharaData.json"
    download_dir = r"D:\UserData\Downloads\Unit2"

    for asset in get_asset_entries(download_dir, manifest):
        download_asset(*asset)


if __name__ == '__main__':
    main()
