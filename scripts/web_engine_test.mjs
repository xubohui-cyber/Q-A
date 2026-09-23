/* 在 Node 里直接跑网页端同一份检索代码（web/js/rag.js），
 * 验证「网页实际取到的结果」与离线 Python 评测结果一致。
 * 需要先启动静态服务器：python -m http.server 8777（web 目录下）
 */
import { readFileSync } from "node:fs";
import { Rag, extractMetric, fmtYi, composeExtractive } from "../web/js/rag.js";

const BASE = process.env.WEB_BASE || "http://127.0.0.1:8777/";
const root = new URL("../", import.meta.url).pathname;
// 与离线端「纯 BM25」结果对比，隔离出「引擎一致性」这一个变量
const raw = JSON.parse(readFileSync(root + "data/out/eval_raw_bm25.json", "utf8"));

const rag = new Rag({ dataBase: new URL("data/", BASE).href });
const t0 = Date.now();
await rag.loadBase(({ stage, ratio }) => {
  if (ratio !== undefined) process.stdout.write(`\r  ${stage} ${Math.round((ratio || 0) * 100)}%   `);
});
console.log(`\n索引加载完成：${rag.chunks.length} 块，${((Date.now() - t0) / 1000).toFixed(1)}s`);

let sameTop = 0;
let total = 0;
for (const q of raw.questions) {
  if (q.panorama) continue;
  const res = await rag.search(q.question, {
    topK: q.opts.top_k,
    alpha: 0.5,
    mode: "bm25",
    code: q.opts.code || null,
    period: q.opts.period || null,
    autoFilter: false,
  });
  const jsIds = res.results.map((c) => c.idx);
  // 离线评测里记的是索引位置；用公司+页码+文本前 30 字定位比对
  const pySig = q.retrieved.map((r) => `${r.company}|${r.period}|${r.page}|${r.type}`);
  const jsSig = res.results.map((c) => `${c.name}|${c.report.period}|${c.pageStart}|${c.type}`);
  total += 1;
  const ok = pySig.join(",") === jsSig.join(",");
  if (ok) sameTop += 1;
  console.log(`${ok ? "✔" : "✘"} ${q.id}  网页 Top1: ${jsSig[0]}  | 离线 Top1: ${pySig[0]}`);
  if (!ok) {
    console.log("   网页:", jsSig.join(" > "));
    console.log("   离线:", pySig.join(" > "));
  }
}
console.log(`\n单公司题 Top-K 与离线一致：${sameTop}/${total}`);

// 全景模式：对比网页解析出的指标与离线
for (const q of raw.questions.filter((x) => x.panorama)) {
  const out = await rag.searchGrouped(q.question, { perCompany: 3, period: q.opts.period });
  const rows = out.groups.map((g) => {
    const text = g.results.map((c) => c.text).join("\n");
    const rev = extractMetric(text, "revenue");
    const pro = extractMetric(text, "profit");
    return { company: g.company.name, rev: rev?.now ?? null, pro: pro?.now ?? null };
  });
  const py = new Map(q.panorama_table.map((r) => [r.company, r]));
  let diff = 0;
  for (const r of rows) {
    const p = py.get(r.company);
    if (!p) continue;
    if (Math.abs((p.revenue ?? -1) - (r.rev ?? -1)) > 1 || Math.abs((p.profit ?? -1) - (r.pro ?? -1)) > 1) {
      diff += 1;
      console.log(`  差异 ${q.id} ${r.company}: 网页 ${fmtYi(r.rev)}/${fmtYi(r.pro)} vs 离线 ${fmtYi(p.revenue)}/${fmtYi(p.profit)}`);
    }
  }
  const top3 = rows
    .filter((r) => r.rev !== null)
    .sort((a, b) => b.rev - a.rev)
    .slice(0, 5)
    .map((r) => `${r.company} ${fmtYi(r.rev)}`);
  console.log(`${diff === 0 ? "✔" : "✘"} ${q.id} 覆盖 ${rows.length} 家，差异 ${diff} 处；营收前五：${top3.join("、")}`);
}
