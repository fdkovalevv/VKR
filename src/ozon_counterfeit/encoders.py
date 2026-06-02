from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import ensure_dir, resolve_torch_device


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for display only.
    tqdm = None


def load_cached_array(cache_path: str | Path) -> np.ndarray | None:
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)
    return None


def save_cached_array(cache_path: str | Path, array: np.ndarray) -> np.ndarray:
    cache_path = Path(cache_path)
    ensure_dir(cache_path.parent)
    np.save(cache_path, array.astype(np.float32))
    return array.astype(np.float32)


class FrozenTextEncoder:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        from sentence_transformers import SentenceTransformer

        self.device = resolve_torch_device(device)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


class ImagePathDataset:
    def __init__(self, paths: list[str], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image

        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)


class FrozenImageEncoder:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        import timm
        import torch

        self.device = resolve_torch_device(device)
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        self.model.eval()
        self.model.to(self.device)
        self.embedding_dim = int(getattr(self.model, "num_features", 0))

        data_config = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        self._torch = torch

    def encode(
        self,
        image_paths: list[str],
        batch_size: int = 64,
        num_workers: int = 2,
        desc: str = "Encoding images",
    ) -> np.ndarray:
        from torch.utils.data import DataLoader

        torch = self._torch

        dataset = ImagePathDataset(image_paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device == "cuda",
        )

        embeddings = []
        batch_iterator = loader
        if tqdm is not None:
            batch_iterator = tqdm(loader, desc=desc, leave=False)

        with torch.no_grad():
            for batch in batch_iterator:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                embeddings.append(outputs.detach().cpu().numpy())

        return np.concatenate(embeddings, axis=0).astype(np.float32)
