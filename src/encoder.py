import json
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort

from .data import ROOT, digest


class Encoder:
    def __init__(self, threads=6, max_length=80, folder=None):
        # Для токенизатора PyTorch не нужен, веса исполняются через ONNX Runtime.
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        from transformers import AutoTokenizer

        folder = Path(folder) if folder is not None else ROOT / "artifacts/encoder"
        self.max_length = max_length
        self.weights_key = self.weights_fingerprint(folder)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(folder), local_files_only=True
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(folder / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.inputs = {x.name for x in self.session.get_inputs()}

    @staticmethod
    def weights_fingerprint(folder=None):
        folder = Path(folder) if folder is not None else ROOT / "artifacts/encoder"
        h = hashlib.sha256()
        for name in [
            "model.onnx",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ]:
            h.update(digest(folder / name).encode())
        h.update(digest(Path(__file__)).encode())
        return h.hexdigest()

    @staticmethod
    def fingerprint(folder=None, max_length=80):
        return f"{Encoder.weights_fingerprint(folder)}:{max_length}:mean:l2"

    @property
    def signature(self):
        return f"{self.weights_key}:{self.max_length}:mean:l2"

    def encode(self, texts, prefix, batch_size=128):
        vectors = []
        texts = [prefix + t if not t.startswith(prefix) else t for t in texts]
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            feed = {
                key: tokens[key].astype(np.int64)
                for key in ["input_ids", "attention_mask"]
            }
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = tokens.get(
                    "token_type_ids", np.zeros_like(feed["input_ids"])
                ).astype(np.int64)
            hidden = self.session.run(None, feed)[0]
            mask = tokens["attention_mask"][:, :, None].astype(np.float32)
            mean = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            vectors.append(
                (
                    mean
                    / np.clip(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12, None)
                ).astype(np.float32)
            )
        return np.vstack(vectors) if vectors else np.empty((0, 384), np.float32)


def item_text(items):
    titles = items.item_title_raw.str[:200]
    params = items.item_infm_params_text.str[:200]
    return [
        f"{title} {param}".strip() if param else title
        for title, param in zip(titles, params)
    ]


def embedding_key(texts, prefix, encoder=None, max_length=80):
    encoder_key = (
        encoder.signature
        if encoder is not None
        else Encoder.fingerprint(max_length=max_length)
    )
    return hashlib.sha256(
        json.dumps([encoder_key, prefix, texts], ensure_ascii=False).encode()
    ).hexdigest()


def embeddings(
    texts, prefix, path, encoder=None, *, bundle=None, rebuild=True, max_length=80
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = embedding_key(texts, prefix, encoder, max_length)
    if not rebuild and not path.exists():
        path = Path(bundle or path)
    meta = path.with_suffix(".key")
    if path.exists() and meta.exists():
        saved = meta.read_text().splitlines()
        if saved == [signature, digest(path)]:
            return np.load(path, mmap_mode="r")
    if not rebuild:
        raise ValueError(
            f"Изменились тексты, encoder или параметры для {path.name}. Включите REBUILD."
        )
    encoder = encoder or Encoder(max_length=max_length)
    codes, unique = pd.factorize(pd.Series(texts), sort=False)
    temporary = path.with_suffix(".part.npy")
    progress = path.with_suffix(".progress.json")
    done = 0
    if temporary.exists() and progress.exists():
        state = json.loads(progress.read_text())
        if state["text_sha256"] == signature:
            done = state["done"]
    vectors = np.lib.format.open_memmap(
        temporary,
        mode="r+" if done else "w+",
        dtype=np.float32,
        shape=(len(unique), 384),
    )
    for start in range(done, len(unique), 4096):
        end = min(start + 4096, len(unique))
        vectors[start:end] = encoder.encode(unique[start:end].tolist(), prefix)
        vectors.flush()
        progress.write_text(json.dumps({"text_sha256": signature, "done": end}))
        if start % 32768 == 0:
            print("embedding", end, "/", len(unique), flush=True)
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(texts), 384)
    )
    for start in range(0, len(texts), 8192):
        output[start : start + 8192] = vectors[codes[start : start + 8192]]
    output.flush()
    del output, vectors
    meta.write_text(signature + "\n" + digest(path) + "\n")
    temporary.unlink()
    progress.unlink()
    return np.load(path, mmap_mode="r")
