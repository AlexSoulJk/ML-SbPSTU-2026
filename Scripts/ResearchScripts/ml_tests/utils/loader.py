from pathlib import Path


def load_metadata_from_folder(folder_path):
    metadata_file = Path(folder_path) / "metadata" / "images_metadata.json"
    if metadata_file.exists():
        import json
        with open(metadata_file, "r", encoding="utf-8") as f:
            images_info = json.load(f)
        return images_info
    else:
        print(f"Metadata file not found: {metadata_file}")
        return None