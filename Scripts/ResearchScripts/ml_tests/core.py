import numpy as np
import torch
from PIL import Image
from ml_tests.init_model import device, preprocess, model


def get_embedding(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image)

        # cosine-friendly normalized embedding
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()[0]

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