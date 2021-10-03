import json

MANIFEST_JSON_PATH = "../download-all-assets/manifest.json"
EXPORT_PATH = "manifest-names.json"


def main():
    with open(MANIFEST_JSON_PATH, encoding="utf-8") as f:
        sample = json.load(f)

    with open(EXPORT_PATH, encoding="utf-8") as f:
        for cat in sample["categories"]:
            for asset in cat["assets"]:
                hash_ = asset["hash"]
                hash_dir = hash_[:2]

                f.write(f"{hash_dir}\\{hash_} - {asset['name']}\n")


if __name__ == '__main__':
    main()
