"""构建 BM25 倒排索引 + 向量索引（bge-small-zh-v1.5 / ONNX）。

输出（work/data/out/）：
  bm25.pkl      倒排表、idf、文档长度
  dense.npy     归一化后的句向量（float32，行与 chunks.jsonl 一一对应）
  index_meta.json
"""

import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenize import tokenize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "out")
CHUNKS = os.path.join(OUT, "chunks.jsonl")
# 模型只存一份，放在网页目录下，离线脚本与浏览器共用
MODEL_DIR = os.path.join(ROOT, "model")

K1, B = 1.5, 0.75
MAX_LEN = 512
MIN_DF = 2  # 只出现 1 次的词项对召回帮助很小，却在网页端占用大量索引体积


def load_chunks():
    with open(CHUNKS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def bm25_text(c):
    """BM25 索引文本：拼上章节上下文，便于「管理层讨论与分析」这类章节性提问。"""
    return " ".join(x for x in [c["section"], c["subsection"], c["text"]] if x)


def build_bm25(chunks):
    postings = defaultdict(dict)  # term -> {doc: tf}
    lens = np.zeros(len(chunks), dtype=np.float32)
    for i, c in enumerate(chunks):
        toks = tokenize(bm25_text(c))
        lens[i] = len(toks)
        for t, tf in Counter(toks).items():
            postings[t][i] = tf
    n = len(chunks)
    if MIN_DF > 1:
        postings = defaultdict(dict, {t: p for t, p in postings.items() if len(p) >= MIN_DF})
    avgdl = float(lens.mean())
    idf = {
        t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()
    }
    return {
        "postings": {t: sorted(p.items()) for t, p in postings.items()},
        "idf": idf,
        "lens": lens,
        "avgdl": avgdl,
        "k1": K1,
        "b": B,
        "n": n,
    }


def embed_texts(texts, batch_size=32, model_name="onnx/model_quantized.onnx"):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_LEN)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, model_name), providers=["CPUExecutionProvider"]
    )
    names = {i.name for i in sess.get_inputs()}
    out_name = sess.get_outputs()[0].name
    vecs = []
    t0 = time.time()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tok.encode_batch(batch)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = sess.run([out_name], feed)[0]
        cls = hidden[:, 0]  # bge 系列用 [CLS] 池化
        cls = cls / np.linalg.norm(cls, axis=1, keepdims=True)
        vecs.append(cls.astype(np.float32))
        done = min(start + batch_size, len(texts))
        if start % (batch_size * 10) == 0 or done == len(texts):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"    embedding {done}/{len(texts)}  {rate:.1f} 块/秒", flush=True)
    return np.vstack(vecs)


def embed_input(c):
    """送入向量的文本：带上公司 / 报告期 / 章节，帮助语义匹配。"""
    ctx = " ".join(x for x in [c["name"], c["period"], c["section"], c["subsection"]] if x)
    return f"{ctx}\n{c['text']}"[:1500]


def main(bm25_only=False, dense_only=False):
    chunks = load_chunks()
    print(f"块数 {len(chunks)}")
    bm = None
    if not dense_only:
        t0 = time.time()
        bm = build_bm25(chunks)
        print(f"BM25 词项 {len(bm['idf'])}（df>={MIN_DF}），耗时 {time.time()-t0:.1f}s", flush=True)
        with open(os.path.join(OUT, "bm25.pkl"), "wb") as f:
            pickle.dump(bm, f)
    if bm25_only:
        return
    if bm is None:
        with open(os.path.join(OUT, "bm25.pkl"), "rb") as f:
            bm = pickle.load(f)

    texts = [embed_input(c) for c in chunks]
    vecs = embed_texts(texts)
    np.save(os.path.join(OUT, "dense.npy"), vecs)
    meta = {
        "n_chunks": len(chunks),
        "dim": int(vecs.shape[1]),
        "bm25_k1": K1,
        "bm25_b": B,
        "avgdl": bm["avgdl"],
        "model": "BAAI/bge-small-zh-v1.5 (Xenova ONNX int8)",
        "chunker": {"target": 700, "min": 220, "overlap": "1 段"},
    }
    json.dump(meta, open(os.path.join(OUT, "index_meta.json"), "w"), ensure_ascii=False, indent=2)
    print("向量维度", vecs.shape, "->", OUT)


if __name__ == "__main__":
    main(bm25_only="--bm25-only" in sys.argv, dense_only="--dense-only" in sys.argv)
