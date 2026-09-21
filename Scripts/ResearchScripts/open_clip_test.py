from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "ViT-B-32"
CACHE_DIR = Path(r"C:\Users\Hp\OneDrive\Рабочий стол\учёба\ML\ML-SbPSTU-2026\checkpoints")
MODEL_PATH = r"C:\Users\Hp\OneDrive\Рабочий стол\учёба\ML\ML-SbPSTU-2026\checkpoints\models--chendelong--RemoteCLIP\snapshots\bf1d8a3ccf2ddbf7c875705e46373bfe542bce38\RemoteCLIP-ViT-B-32.pt"

# 1. Скачиваем RemoteCLIP checkpoint.
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


def get_embedding(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image)

        # cosine-friendly normalized embedding
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()[0]

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

def group_by_samples_and_period(images_info):
    grouped_data = {}
    for image in images_info.values():
        sample_id = image["sample_id"]
        period = image["period"]
        if sample_id not in grouped_data:
            grouped_data[sample_id] = {}
        if period not in grouped_data[sample_id]:
            grouped_data[sample_id][period] = []
        grouped_data[sample_id][period].append(image)
    return grouped_data

def embedding_consistency_loo(embeddings):
    X = np.stack(embeddings)

    result = []

    for i in range(len(X)):
        if len(X) == 1:
            result.append(np.nan)
            continue

        others = np.delete(X, i, axis=0)

        center = others.mean(axis=0)
        center /= np.linalg.norm(center)

        sim = float(np.dot(X[i], center))
        result.append(sim)

    return np.array(result)

def aggregate_embeddings_by_sample_and_period(embeddings_for_period):
    X = np.stack(embeddings_for_period)
    z = X.mean(axis=0)
    z /= np.linalg.norm(z)
    similarities = embedding_consistency_loo(embeddings_for_period)
    return z, similarities

def calculate_cosine_similarity(embedding1, embedding2):
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity
    
def get_butch_embeddings(images_info_grouped):
    embeddings_pre = []
    embeddings_post = []
    labels_pre = []
    labels_post = []
    statistic_info = {"cosine_similarity_at_one_point": {}}
    for sample_id, periods in images_info_grouped.items():
        sample_statistic = statistic_info["cosine_similarity_at_one_point"].get(sample_id, None)
        
        if sample_statistic is None:
            statistic_info["cosine_similarity_at_one_point"][sample_id] = {}
            
        for period, images in periods.items():
            embeddings_for_period = []
            
            sample_statistic_period = statistic_info["cosine_similarity_at_one_point"][sample_id].get(period, None)
                    
            if sample_statistic_period is None:
                statistic_info["cosine_similarity_at_one_point"][sample_id][period] = {}
            
            for image in images:
                embedding = get_embedding(image["path"])
                embeddings_for_period.append(embedding)
            
            aggregate_embedding, similarities = aggregate_embeddings_by_sample_and_period(embeddings_for_period)
            
            if period == "PRE":
                embeddings_pre.append(aggregate_embedding)
                labels_pre.append(images[0]["label"])
                
                
            elif period == "POST":
                embeddings_post.append(aggregate_embedding)
                labels_post.append(images[0]["label"])
                
            statistic_info["cosine_similarity_at_one_point"][sample_id][period] = \
            (similarities.tolist(), list(map(lambda x: x["image_id"], images)))
            
            
    return embeddings_pre, labels_pre, embeddings_post, labels_post, statistic_info

def plot_pca(X, labels, title):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    print(
        title,
        "explained variance:",
        pca.explained_variance_ratio_,
        "sum:",
        pca.explained_variance_ratio_.sum()
    )

    plt.figure()

    for label in sorted(set(labels)):
        mask = labels == label
        plt.scatter(
            X2[mask, 0],
            X2[mask, 1],
            label=label
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.show()

def class_similarity(X, labels):
    sim = cosine_similarity(X)

    same = []
    different = []

    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if labels[i] == labels[j]:
                same.append(sim[i, j])
            else:
                different.append(sim[i, j])

    print(
        "same class:",
        np.mean(same),
        "+/-",
        np.std(same),
    )

    print(
        "different class:",
        np.mean(different),
        "+/-",
        np.std(different),
    )

def print_statistics(statistic_info):
    for sample_id, periods in statistic_info["cosine_similarity_at_one_point"].items():
        print(f"Sample ID: {sample_id}")
        for period, similarities in periods.items():
            print(f"  Period: {period}")
            for similarity, image_ids in zip(*similarities):
                print(f"    Similarity: {similarity}, Image IDs: {image_ids}")

def similarity_gap(X, labels):
    labels = np.asarray(labels)

    sim = cosine_similarity(X)

    same = []
    different = []

    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if labels[i] == labels[j]:
                same.append(sim[i, j])
            else:
                different.append(sim[i, j])

    return np.mean(same) - np.mean(different)


def permutation_test(X, labels, n_perm=10_000):
    labels = np.asarray(labels)

    observed = similarity_gap(X, labels)

    rng = np.random.default_rng(42)

    random_scores = np.empty(n_perm)

    for i in range(n_perm):
        shuffled_labels = rng.permutation(labels)

        random_scores[i] = similarity_gap(
            X,
            shuffled_labels,
        )

    p_value = (
        np.sum(random_scores >= observed) + 1
    ) / (n_perm + 1)

    print("Observed gap:", observed)
    print(
        "Random:",
        random_scores.mean(),
        "+/-",
        random_scores.std()
    )
    print("p-value:", p_value)

    return observed, random_scores

def permutation_test_post_vs_pre(
    X_pre,
    X_post,
    labels,
    n_perm=10_000,
):
    labels = np.asarray(labels)

    observed = (
        similarity_gap(X_post, labels)
        - similarity_gap(X_pre, labels)
    )

    rng = np.random.default_rng(42)
    random_scores = np.empty(n_perm)

    for i in range(n_perm):
        shuffled = rng.permutation(labels)

        random_scores[i] = (
            similarity_gap(X_post, shuffled)
            - similarity_gap(X_pre, shuffled)
        )

    p_value = (
        np.sum(random_scores >= observed) + 1
    ) / (n_perm + 1)

    print("POST - PRE observed:", observed)
    print(
        "Random:",
        random_scores.mean(),
        "+/-",
        random_scores.std(),
    )
    print("p-value:", p_value)

    return observed, random_scores

def permutation_test_post_vs_pre(
    X_pre,
    X_post,
    labels,
    n_perm=10_000,
):
    labels = np.asarray(labels)

    observed = (
        similarity_gap(X_post, labels)
        - similarity_gap(X_pre, labels)
    )

    rng = np.random.default_rng(42)
    random_scores = np.empty(n_perm)

    for i in range(n_perm):
        shuffled = rng.permutation(labels)

        random_scores[i] = (
            similarity_gap(X_post, shuffled)
            - similarity_gap(X_pre, shuffled)
        )

    p_value = (
        np.sum(random_scores >= observed) + 1
    ) / (n_perm + 1)

    print("POST - PRE observed:", observed)
    print(
        "Random:",
        random_scores.mean(),
        "+/-",
        random_scores.std(),
    )
    print("p-value:", p_value)

    return observed, random_scores

TRESHHOLDERS = {}

PATH_TO_IMAGES_DATA = r"Scripts\ResearchScripts\Data\Images"

images_info = load_metadata_from_folder(PATH_TO_IMAGES_DATA)
grouped_data = group_by_samples_and_period(images_info)
embeddings_pre, labels_pre, embeddings_post, labels_post, statistic_info = get_butch_embeddings(grouped_data)
# print(labels_pre)
# print(labels_post)
# print_statistics(statistic_info)

plot_pca(np.array(embeddings_pre), np.array(labels_pre), "PCA of aggregated embeddings by sample and period PRE")
plot_pca(np.array(embeddings_post), np.array(labels_post), "PCA of aggregated embeddings by sample and period")
plot_pca(np.array(embeddings_post) - np.array(embeddings_pre), np.array(labels_post), "PCA of aggregated embeddings by sample and period")
# class_similarity(np.array(embeddings_pre), np.array(labels_pre))
# class_similarity(np.array(embeddings_post), np.array(labels_post))
# class_similarity(np.array(embeddings_post) - np.array(embeddings_pre), np.array(labels_post))

### Подозрительные товарищи.
# | Sample ID                                  | Period | Image ID                           |        Sim |
# | ------------------------------------------ | ------ | ---------------------------------- | ---------: |
# | `3bf38f91472bef021ff29a3a4a38d269a10d29d8` | POST   | `3bfc21d690359dddf72f6666b60beca4` | **0.7384** |
# | `a59040b8c56f0c2d9837ceda8fb921602df5e7fb` | POST   | `4ab87c59318743e15b1ea5f6dcee005c` | **0.8017** |
# | `3bf38f91472bef021ff29a3a4a38d269a10d29d8` | PRE    | `ee0509de3a1f0c33cfdc2f8ce342f087` | **0.8091** |
# | `2c33ffed48d98c7ec377222147a5e42e46eda320` | PRE    | `1a0d072f779a205f6e0763e837e38a3a` | **0.8128** |
# | `1971d07068b48f960017c168d620ceaea1598ec3` | PRE    | `114cec2c99f669a69e38c5a0287a3330` | **0.8144** |
# | `ca8935d27317a17de0c152bd7481d7051afd0e6b` | PRE    | `6f27a950aa3fa20d41fde63a50ec14f2` | **0.8231** |
# | `f6ed4acb9d9691f0f1cc3373b4ba7d0c31f04c6a` | POST   | `4691a09b16d2f131c9e77f7f7168d0ba` | **0.8268** |
# | `160683842cfa33ec5b9ba1d772b08a05e904b8a4` | POST   | `f812ee2e87464213c2f627f09d886367` | **0.8381** |
# | `e27c94bc888f7f4e7fa4135b35cc8a76384fbed1` | PRE    | `aeed0b9cec4c6b5dd14cf9536e802368` | **0.8429** |
# | `cc8a64b02d9a3e39110815c82289f50a53f1aa5c` | POST   | `d31c245e23f5c96a48022b682b17f1ad` | **0.8480** |

print("PRE")
permutation_test(embeddings_pre, labels_pre)

print("POST")
permutation_test(embeddings_post, labels_post)

print("DELTA")
permutation_test(np.array(embeddings_post) - np.array(embeddings_pre), np.array(labels_post))

# PRE
# Observed gap: 0.010017574
# Random: -2.863237261772156e-05 +/- 0.004952480882259092
# p-value: 0.030596940305969402
# POST
# Observed gap: 0.026970744
# Random: 4.26986038684845e-05 +/- 0.004650981699400781
# p-value: 9.999000099990002e-05
# DELTA
# Observed gap: 0.040752493
# Random: 2.8531200997531416e-05 +/- 0.011656466695485557
# p-value: 0.0037996200379962005

observed, random_scores = permutation_test_post_vs_pre(
    np.array(embeddings_pre),
    np.array(embeddings_post),
    np.array(labels_post)
)

# print("POST - PRE observed:", observed)
# print(
#     "Random:",
#     random_scores.mean(),
#     "+/-",
#     random_scores.std(),
# )
# print("p-value:", (np.sum(random_scores >= observed) + 1) / (len(random_scores) + 1))

# POST - PRE observed: 0.01695317
# Random: 7.133097648620606e-05 +/- 0.005932764105947036
# p-value: 0.006399360063993601

