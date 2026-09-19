from functools import lru_cache

import numpy as np
import torch
import open_clip

from config import MODEL_NAME, PRETRAINED


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_model():
    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    return model, preprocess, tokenizer, device


def get_preprocess():
    return get_model()[1]


def get_tokenizer():
    return get_model()[2]


def _run_under_autocast(fn, device):
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return fn()
    return fn()


def embed_text(text):
    model, _, tokenizer, device = get_model()
    tokens = tokenizer(text)

    def _run():
        with torch.no_grad():
            features = model.encode_text(tokens.to(device))
            return torch.nn.functional.normalize(features, dim=-1)

    features = _run_under_autocast(_run, device)
    return features.cpu().numpy().astype(np.float32)


def embed_pil_images(images, batch_size=32):
    model, preprocess, _, device = get_model()
    if not images:
        return np.zeros((0, model.visual.output_dim), dtype=np.float32)

    outputs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            tensors = torch.stack([preprocess(im) for im in images[i : i + batch_size]]).to(device)

            def _run():
                features = model.encode_image(tensors)
                return torch.nn.functional.normalize(features, dim=-1)

            features = _run_under_autocast(_run, device)
            outputs.append(features.cpu().numpy().astype(np.float32))

    return np.concatenate(outputs, axis=0)