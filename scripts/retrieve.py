"""混合检索：BM25（字符 bigram + 英文词）+ 向量（bge-small-zh-v1.5）分数融合。

支持：
* 按公司 / 报告期过滤（问句里出现公司名或代码时自动启用）；
* 全景题的分组召回（每家公司各取 top-n，再汇总）。
"""

import json
import math
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenize import tokenize  # noqa: E402
from query_rewrite import expand_query  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "out")
MODEL_DIR = os.path.join(ROOT, "web", "model")


def minmax(d):
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi <= lo:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


class Retriever:
    def __init__(self):
        self.chunks = [
            json.loads(l) for l in open(os.path.join(OUT, "chunks.jsonl"), encoding="utf-8")
        ]
        with open(os.path.join(OUT, "bm25.pkl"), "rb") as f:
            self.bm = pickle.load(f)
        self.dense = np.load(os.path.join(OUT, "dense.npy"))
        self.k1, self.b, self.avgdl = self.bm["k1"], self.bm["b"], self.bm["avgdl"]
        self.companies = {}
        for c in self.chunks:
            self.companies.setdefault(c["code"], c["name"])
        self.name2code = {v: k for k, v in self.companies.items()}
        self._sess = None
        self._tok = None

    # ---------- 向量 ----------
    def _model(self):
        if self._sess is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
            self._tok.enable_truncation(max_length=512)
            self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
            self._sess = ort.InferenceSession(
                os.path.join(MODEL_DIR, "onnx/model_quantized.onnx"),
                providers=["CPUExecutionProvider"],
            )
        return self._sess

    def embed(self, texts):
        sess = self._model()
        names = {i.name for i in sess.get_inputs()}
        enc = self._tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = sess.run(None, feed)[0][:, 0]
        return hidden / np.linalg.norm(hidden, axis=1, keepdims=True)

    # ---------- BM25 ----------
    def bm25(self, query, allowed=None):
        scores = {}
        post, idf = self.bm["postings"], self.bm["idf"]
        for t in set(tokenize(query)) | set(tokenize(expand_query(query))):
            tf_docs = post.get(t)
            if not tf_docs:
                continue
            w = idf[t]
            for doc, tf in tf_docs:
                if allowed is not None and doc not in allowed:
                    continue
                dl = self.bm["lens"][doc]
                s = w * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                scores[doc] = scores.get(doc, 0.0) + s
        return scores

    # ---------- 过滤 ----------
    def detect_filter(self, query, code=None, period=None, kind=None):
        if code is None:
            for c, n in self.companies.items():
                if n in query or (c in query):
                    code = c
                    break
        if code is None:  # 简称 / 别名
            alias = {
                "茅台": "600519", "洋河": "002304", "汾酒": "600809", "老窖": "000568",
                "古井": "000596", "迎驾": "603198", "口子": "603589", "舍得": "600702",
                "水井坊": "600779", "金徽": "603919", "酒鬼": "000799", "天佑德": "002646",
                "金种子": "600199", "今世缘": "603369", "老白干": "600559", "五粮液": "000858",
            }
            for k, v in alias.items():
                if k in query:
                    code = v
                    break
        if period is None:
            if "半年" in query or "中报" in query or "上半年" in query:
                period = "H1"
            elif "2024" in query:
                period = "2024A"
            elif "2025" in query:
                period = "2025A"
        return code, period, kind

    def allowed_set(self, code=None, period=None, kind=None):
        ids = set()
        for i, c in enumerate(self.chunks):
            if code and c["code"] != code:
                continue
            if period and period not in c["period"]:
                continue
            if kind and c["kind"] != kind:
                continue
            ids.add(i)
        return ids

    # ---------- 主检索 ----------
    def search(self, query, top_k=8, alpha=0.5, code=None, period=None, kind=None,
               auto_filter=True, return_scores=True, mode="hybrid"):
        if auto_filter and code is None and period is None:
            code, period, kind = self.detect_filter(query, code, period, kind)
        allowed = self.allowed_set(code, period, kind)
        if not allowed:  # 过滤太严就退化为全库
            allowed = None
            code = period = kind = None
        bm = self.bm25(query, allowed)
        if mode == "bm25":
            dn = {}
        else:
            qv = self.embed([query])[0]
            idx = np.fromiter(allowed, dtype=np.int64) if allowed else np.arange(len(self.chunks))
            cos = self.dense[idx] @ qv
            dn = {int(i): float(s) for i, s in zip(idx, cos)}
        bm_n, dn_n = minmax(bm), minmax(dn)
        fused = {}
        if mode == "bm25":
            fused = dict(bm_n)
        elif mode == "dense":
            fused = dict(dn_n)
        else:
            for i in set(bm_n) | set(dn_n):
                fused[i] = alpha * dn_n.get(i, 0.0) + (1 - alpha) * bm_n.get(i, 0.0)
        # 并列分数用文档编号做 tie-break，保证与网页端（JS）排序完全一致
        order = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
        results = []
        for rank, i in enumerate(order, 1):
            c = dict(self.chunks[i])
            c["idx"] = i
            c["rank"] = rank
            c["score"] = round(fused[i], 4)
            if return_scores:
                c["bm25"] = round(bm.get(i, 0.0), 3)
                c["cos"] = round(dn.get(i, 0.0), 4)
            results.append(c)
        return {
            "query": query,
            "filter": {"code": code, "period": period, "kind": kind},
            "bm25_top": sorted(bm, key=lambda i: (-bm[i], i))[:top_k],
            "dense_top": sorted(dn, key=lambda i: (-dn[i], i))[:top_k],
            "results": results,
            "alpha": alpha,
        }

    def metric_query(self, question):
        """全景题改写：把口语问题换成「主要会计数据 + 指标名」，更容易命中可比口径的表。"""
        q = re.sub(r"^\s*全景\s*[:：]?\s*", "", question)
        if "营业收入" in q or "营收" in q:
            return "主要会计数据 营业收入 上年同期 增减"
        if "净利润" in q:
            return "主要会计数据 归属于上市公司股东的净利润 上年同期 增减"
        if "毛利率" in q:
            return "主营业务 分产品 毛利率 比上年同期增减"
        return q

    def search_grouped(self, query, per_company=2, alpha=0.5, period=None, kind=None):
        """全景题：先做一次广域检索，再按公司分组，保证每家公司都有代表证据。"""
        q = self.metric_query(query)
        wide = max(per_company * len(self.companies) * 4, 64)
        res = self.search(q, top_k=wide, alpha=alpha, period=period, kind=kind,
                          auto_filter=False)
        groups = {}
        for c in res["results"]:
            g = groups.setdefault(c["code"], [])
            if len(g) < per_company:
                g.append(c)
        for code, name in sorted(self.companies.items()):
            if code in groups:
                continue
            one = self.search(q, top_k=per_company, alpha=alpha, code=code,
                              period=period, kind=kind, auto_filter=False)
            if one["results"]:
                groups[code] = one["results"]
        out = {
            code: {"name": self.companies[code], "results": items}
            for code, items in sorted(groups.items())
        }
        return {"query": q, "orig_query": query, "groups": out, "per_company": per_company}


if __name__ == "__main__":
    r = Retriever()
    for q in sys.argv[1:] or ["贵州茅台2025年营业收入是多少"]:
        res = r.search(q, top_k=5)
        print("Q:", q, "filter:", res["filter"])
        for c in res["results"]:
            print(f"  [{c['rank']}] {c['name']} {c['period']} {c['section']} P{c['page_start']}"
                  f" {c['type']} score={c['score']} bm25={c['bm25']} cos={c['cos']}")
            print("       ", c["text"][:110].replace("\n", " / "))
