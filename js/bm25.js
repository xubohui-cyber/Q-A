/* BM25 检索：字符 bigram + 英文词的倒排索引（与离线 Python 端同一套分词规则） */

import { normalizeBert } from "./tokenizer.js";

const CJK = "\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff";
const RUN_RE = new RegExp(`[${CJK}]+|[a-z0-9][a-z0-9.%,]*|[\\u00c0-\\u024f]+`, "g");

/** 与 Python scripts/tokenize.py 完全一致的词项切分 */
export function tokenize(text) {
  const s = (text || "").normalize("NFKC").toLowerCase();
  const out = [];
  for (const m of s.matchAll(RUN_RE)) {
    const tok = m[0];
    const first = tok.codePointAt(0);
    const isCjk = (first >= 0x4e00 && first <= 0x9fff) || (first >= 0x3400 && first <= 0x4dbf);
    if (!isCjk) {
      out.push(tok);
      continue;
    }
    for (let i = 0; i < tok.length; i++) out.push(tok[i]);
    for (let i = 0; i < tok.length - 1; i++) out.push(tok.slice(i, i + 2));
  }
  return out;
}

function readVarint(buf, pos) {
  let result = 0;
  let shift = 0;
  while (true) {
    const b = buf[pos.i++];
    result += (b & 0x7f) * Math.pow(2, shift);
    if ((b & 0x80) === 0) return result;
    shift += 7;
  }
}

export class BM25Index {
  constructor(buffer) {
    const u8 = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const magic = String.fromCharCode(...u8.subarray(0, 4));
    if (magic !== "BM25") throw new Error("BM25 索引格式不正确");
    this.n = dv.getUint32(4, true);
    this.nTerms = dv.getUint32(8, true);
    this.avgdl = dv.getFloat32(12, true);
    this.k1 = dv.getFloat32(16, true);
    this.b = dv.getFloat32(20, true);
    const dictLen = dv.getUint32(24, true);
    this.dictStart = 28;
    this.postStart = this.dictStart + dictLen;
    this.bytes = u8;
    this.decoder = new TextDecoder("utf-8");
    this.dict = new Map(); // term -> [df, offset]
    const pos = { i: this.dictStart };
    for (let t = 0; t < this.nTerms; t++) {
      const len = readVarint(this.bytes, pos);
      const term = this.decoder.decode(this.bytes.subarray(pos.i, pos.i + len));
      pos.i += len;
      const df = readVarint(this.bytes, pos);
      const off = readVarint(this.bytes, pos);
      this.dict.set(term, [df, off]);
    }
  }

  idf(df) {
    return Math.log(1 + (this.n - df + 0.5) / (df + 0.5));
  }

  /** 返回 Map<docId, score>，allowed 为 Set<docId> 时只在其中打分 */
  score(query, allowed = null, lens = null) {
    const scores = new Map();
    const terms = new Set(tokenize(query));
    for (const term of terms) {
      const hit = this.dict.get(term);
      if (!hit) continue;
      const [df, off] = hit;
      const w = this.idf(df);
      const pos = { i: this.postStart + off };
      let doc = 0; // 编码端以 0 为起点做增量编码
      for (let k = 0; k < df; k++) {
        doc += readVarint(this.bytes, pos);
        const tf = readVarint(this.bytes, pos);
        if (allowed && !allowed.has(doc)) continue;
        const dl = lens[doc];
        const denom = tf + this.k1 * (1 - this.b + (this.b * dl) / this.avgdl);
        const s = (w * tf * (this.k1 + 1)) / denom;
        scores.set(doc, (scores.get(doc) || 0) + s);
      }
    }
    return scores;
  }
}
