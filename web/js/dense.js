/* 浏览器端向量检索：onnxruntime-web 跑 bge-small-zh-v1.5（int8 ONNX）编码查询，
 * 再和离线算好、随页面下发的 int8 文档向量做内积。
 */

import { BertTokenizer } from "./tokenizer.js";

let ort = null;

/** onnxruntime-web 只在真正启用语义检索时动态加载，首屏不必下载 14MB wasm */
async function loadOrt(wasmPaths) {
  if (!ort) {
    ort = await import("../vendor/ort/ort.min.mjs");
    ort.env.wasm.wasmPaths = wasmPaths;
    ort.env.wasm.numThreads = 1; // 静态托管下没有 COOP/COEP，单线程最稳
    ort.env.logLevel = "error";
  }
  return ort;
}

export const MODEL_SOURCES = {
  local: {
    label: "仓库自带（同源，推荐）",
    onnx: new URL("../model/onnx/model_quantized.onnx", import.meta.url).href,
    tokenizer: new URL("../model/tokenizer.json", import.meta.url).href,
  },
  cdn: {
    label: "jsDelivr CDN 加速",
    onnx: "https://cdn.jsdelivr.net/gh/xubohui-cyber/Q-A@main/web/model/onnx/model_quantized.onnx",
    tokenizer: "https://cdn.jsdelivr.net/gh/xubohui-cyber/Q-A@main/web/model/tokenizer.json",
  },
};

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`下载失败 ${res.status} ${url}`);
  const total = Number(res.headers.get("content-length") || 0);
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onProgress?.(got / total, got, total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

export class DenseEncoder {
  constructor(source = "local", override = null) {
    this.source = source;
    this.override = override;
    this.ready = false;
  }

  async load(onProgress) {
    const src = this.override || MODEL_SOURCES[this.source];
    const ortLib = await loadOrt(
      this.override?.wasmPaths || new URL("../vendor/ort/", import.meta.url).href
    );
    onProgress?.({ stage: "tokenizer", ratio: 0 });
    const tkJson = await (await fetch(src.tokenizer)).json();
    this.tokenizer = new BertTokenizer(tkJson);
    onProgress?.({ stage: "model", ratio: 0 });
    const bytes = await fetchWithProgress(src.onnx, (ratio, got, total) =>
      onProgress?.({ stage: "model", ratio, got, total })
    );
    this.session = await ortLib.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    this.inputNames = this.session.inputNames;
    this.ready = true;
  }

  /** 返回归一化后的 512 维查询向量（Float32Array） */
  async embed(texts) {
    const encs = texts.map((t) => this.tokenizer.encode(t, 512));
    const len = Math.max(...encs.map((e) => e.ids.length));
    const n = encs.length;
    const ids = new BigInt64Array(n * len);
    const mask = new BigInt64Array(n * len);
    const types = new BigInt64Array(n * len);
    encs.forEach((e, r) => {
      for (let c = 0; c < len; c++) {
        const v = c < e.ids.length ? BigInt(e.ids[c]) : 0n;
        ids[r * len + c] = v;
        mask[r * len + c] = c < e.ids.length ? 1n : 0n;
      }
    });
    const feeds = {
      input_ids: new ort.Tensor("int64", ids, [n, len]),
      attention_mask: new ort.Tensor("int64", mask, [n, len]),
    };
    if (this.inputNames.includes("token_type_ids")) {
      feeds.token_type_ids = new ort.Tensor("int64", types, [n, len]);
    }
    const out = await this.session.run(feeds);
    const hidden = out[this.session.outputNames[0]];
    const dim = hidden.dims[2];
    const data = hidden.data; // Float32Array
    const vecs = [];
    for (let r = 0; r < n; r++) {
      const v = new Float32Array(dim);
      let norm = 0;
      for (let d = 0; d < dim; d++) {
        const x = data[r * len * dim + d]; // [CLS] 池化
        v[d] = x;
        norm += x * x;
      }
      norm = Math.sqrt(norm) || 1;
      for (let d = 0; d < dim; d++) v[d] /= norm;
      vecs.push(v);
    }
    return vecs;
  }
}

/** 用 int8 文档向量做全量内积，返回 Float32Array 分数 */
export function dotInt8(queryVec, int8Matrix, dim) {
  const n = int8Matrix.length / dim;
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const base = i * dim;
    for (let d = 0; d < dim; d++) s += int8Matrix[base + d] * queryVec[d];
    scores[i] = s / 127;
  }
  return scores;
}
