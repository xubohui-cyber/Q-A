"""把索引导出成网页可直接加载的静态文件。

web/data/meta.json     公司 / 报告 / 章节字典 + 统计信息
web/data/chunks.json   全部文本块（数组压缩存储，字段用下标引用字典）
web/data/bm25.bin.gz   BM25 倒排索引（varint 二进制，前端 gzip 解压）
web/data/vec.i8.bin.gz int8 量化向量（512 维，L2 已归一化）
"""

import gzip
import json
import os
import pickle
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "out")
WEB = os.path.join(ROOT, "web", "data")


def put_uv(out, n):
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def export_bm25(bm):
    terms = sorted(bm["idf"])
    post = bm["postings"]
    body = bytearray()
    dict_body = bytearray()
    offsets = []
    for t in terms:
        offsets.append(len(body))
        prev = 0
        for doc, tf in post[t]:
            put_uv(body, doc - prev)
            put_uv(body, min(tf, 64))
            prev = doc
    for t, off in zip(terms, offsets):
        tb = t.encode("utf-8")
        put_uv(dict_body, len(tb))
        dict_body += tb
        put_uv(dict_body, len(post[t]))  # df
        put_uv(dict_body, off)
    head = bytearray(b"BM25")
    head += struct.pack("<II", bm["n"], len(terms))
    head += struct.pack("<fff", bm["avgdl"], bm["k1"], bm["b"])
    head += struct.pack("<I", len(dict_body))
    blob = bytes(head) + bytes(dict_body) + bytes(body)
    path = os.path.join(WEB, "bm25.bin.gz")
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(blob)
    return path, len(blob), len(terms), len(body)


def export_vectors(vec):
    q = np.clip(np.round(vec * 127.0), -127, 127).astype(np.int8)
    path = os.path.join(WEB, "vec.i8.bin.gz")
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(q.tobytes())
    return path, q.shape


def export_doclens(bm):
    """BM25 公式需要每块的词数，单独存一个小文件，前端加载后即可离线打分。"""
    lens = np.round(bm["lens"]).astype("<u4")
    path = os.path.join(WEB, "doclen.bin")
    with open(path, "wb") as f:
        f.write(lens.tobytes())
    return path


def main():
    os.makedirs(WEB, exist_ok=True)
    chunks = [json.loads(l) for l in open(os.path.join(OUT, "chunks.jsonl"), encoding="utf-8")]
    bm = pickle.load(open(os.path.join(OUT, "bm25.pkl"), "rb"))
    vec = np.load(os.path.join(OUT, "dense.npy"))

    codes, reports, sections, subs = {}, [], {}, {}
    for c in chunks:
        codes.setdefault(c["code"], {"code": c["code"], "name": c["name"],
                                     "exchange": c["exchange"]})
    codelist = sorted(codes)
    cidx = {c: i for i, c in enumerate(codelist)}
    ridx = {}
    for c in chunks:
        k = (c["code"], c["period"])
        if k not in ridx:
            ridx[k] = len(reports)
            reports.append({
                "code": c["code"], "name": c["name"], "period": c["period"],
                "kind": c["kind"], "title": c["title"], "url": c["url"],
                "date": c["announcement_date"],
            })

    def intern(store, s):
        if s not in store:
            store[s] = len(store)
        return store[s]

    rows = []
    for c in chunks:
        rows.append([
            cidx[c["code"]],
            ridx[(c["code"], c["period"])],
            intern(sections, c["section"]),
            intern(subs, c["subsection"]),
            c["page_start"],
            c["page_end"],
            1 if c["type"] == "table" else 0,
            c["text"],
        ])

    meta = {
        "generated": "2026-09-23",
        "stats": {
            "companies": len(codelist),
            "reports": len(reports),
            "pages": sum(1 for _ in chunks) and None,
            "chunks": len(rows),
            "tables": sum(1 for c in chunks if c["type"] == "table"),
            "chars": sum(c["n_chars"] for c in chunks),
            "dim": int(vec.shape[1]),
            "bm25_terms": len(bm["idf"]),
            "model": "bge-small-zh-v1.5 (ONNX int8)",
        },
        "companies": [codes[c] for c in codelist],
        "reports": reports,
        "sections": [None] * len(sections),
        "subs": [None] * len(subs),
    }
    for s, i in sections.items():
        meta["sections"][i] = s
    for s, i in subs.items():
        meta["subs"][i] = s
    # 报告页数
    TEXT_DIR = os.path.join(ROOT, "data", "text")
    pages = 0
    for fn in os.listdir(TEXT_DIR):
        pages += json.load(open(os.path.join(TEXT_DIR, fn), encoding="utf-8"))["num_pages"]
    meta["stats"]["pages"] = pages

    with open(os.path.join(WEB, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(WEB, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))

    p, raw, nterm, pbytes = export_bm25(bm)
    print(f"bm25: {nterm} 词项, 原始 {raw/1e6:.2f}MB, 倒排 {pbytes/1e6:.2f}MB -> {os.path.getsize(p)/1e6:.2f}MB gz")
    vp, shape = export_vectors(vec)
    print(f"vec : {shape} -> {os.path.getsize(vp)/1e6:.2f}MB gz")
    lp = export_doclens(bm)
    print(f"doclen.bin {os.path.getsize(lp)/1e3:.0f}KB")
    print(f"chunks.json {os.path.getsize(os.path.join(WEB,'chunks.json'))/1e6:.2f}MB, "
          f"meta.json {os.path.getsize(os.path.join(WEB,'meta.json'))/1e6:.2f}MB")


if __name__ == "__main__":
    sys.exit(main())
