
from pathlib import Path

import open_clip
import torch


MODEL_NAME = "ViT-B-32"
CACHE_DIR = Path(r"C:\Users\Hp\OneDrive\Рабочий стол\учёба\ML\ML-SbPSTU-2026\checkpoints")
MODEL_PATH = r"C:\Users\Hp\OneDrive\Рабочий стол\учёба\ML\ML-SbPSTU-2026\checkpoints\models--chendelong--RemoteCLIP\snapshots\bf1d8a3ccf2ddbf7c875705e46373bfe542bce38\RemoteCLIP-ViT-B-32.pt"

# 1. Скачиваем RemoteCLIP checkpoint.
# from huggingface_hub import hf_hub_download
# checkpoint_path = hf_hub_download(
#     repo_id="chendelong/RemoteCLIP",
#     filename=f"RemoteCLIP-{MODEL_NAME}.pt",
#     cache_dir="./checkpoints",
# )

# print("Checkpoint:", checkpoint_path)

# 2. Создаем архитектуру OpenCLIP.
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)

# 3. Загружаем веса RemoteCLIP.
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

result = model.load_state_dict(checkpoint)
print(result)

device = torch.device("cpu")
model = model.to(device).eval()