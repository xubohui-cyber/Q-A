/* 校验浏览器端 BM25 实现（web/js/bm25.js）与离线 Python 版检索结果一致 */
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { BM25Index } from "../web/js/bm25.js";
import { expandQuery } from "../web/js/rag.js";

const root = new URL("../", import.meta.url).pathname;
const bmBytes = gunzipSync(readFileSync(root + "web/data/bm25.bin.gz"));
const buf = bmBytes.buffer.slice(bmBytes.byteOffset, bmBytes.byteOffset + bmBytes.byteLength);
const bm = new BM25Index(buf);
const lens = new Uint32Array(readFileSync(root + "web/data/doclen.bin").buffer.slice(0));
const ref = JSON.parse(readFileSync(root + "data/out/parity_ref.json", "utf8"));

let bad = 0;
for (const { query, python_top } of ref) {
  const scores = bm.score(expandQuery(query), null, lens);
  const jsTop = [...scores.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, 10)
    .map(([i]) => i);
  const same = jsTop.length === python_top.length && jsTop.every((v, i) => v === python_top[i]);
  if (!same) {
    bad++;
    console.log("MISMATCH:", query);
    console.log("  py:", python_top.join(","));
    console.log("  js:", jsTop.join(","));
  }
}
console.log(bad === 0 ? `BM25 排序完全一致（${ref.length} 条查询）` : `${bad}/${ref.length} 条不一致`);
