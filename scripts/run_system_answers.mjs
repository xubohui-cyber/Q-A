/* 用网页端同一份抽取式答案代码（web/js/answer.js）跑评测题，
 * 产出「系统回答」，供人工判定对错。 */
import { readFileSync, writeFileSync } from "node:fs";
import { composeExtractive, extractMetric, fmtYi, queryTerms } from "../web/js/answer.js";

const root = new URL("../", import.meta.url).pathname;
const raw = JSON.parse(readFileSync(root + "data/out/eval_raw.json", "utf8"));
const out = [];

for (const q of raw.questions) {
  if (q.panorama) {
    const t = q.panorama_table || [];
    const sorted = t.slice().sort((a, b) => (b.revenue || -1) - (a.revenue || -1));
    out.push({
      id: q.id,
      question: q.question,
      answer:
        "全景对比（按 2025/H1 指标自动解析）：\n" +
        sorted
          .map(
            (r) =>
              `  ${r.company}：营收 ${fmtYi(r.revenue)}（同比 ${
                r.revenue_yoy ?? "—"
              }%），归母净利润 ${fmtYi(r.profit)}（同比 ${r.profit_yoy ?? "—"}%）`
          )
          .join("\n"),
    });
    continue;
  }
  const results = q.retrieved.map((r) => ({
    rank: r.rank,
    name: r.company,
    section: r.section,
    subsection: r.subsection,
    pageStart: r.page,
    pageEnd: r.page,
    type: r.type,
    text: r.text,
  }));
  const bullets = composeExtractive(q.question, results, 3);
  // 若题干问的是具体指标，额外用指标解析器直接从 Top-3 文本里取数
  const extra = [];
  const joined = results.slice(0, 3).map((r) => r.text).join("\n");
  for (const [metric, label] of [["revenue", "营业收入"], ["profit", "归母净利润"]]) {
    if (q.question.includes("净利润") && metric === "profit" && label) {
      const m = extractMetric(joined, "profit");
      if (m) extra.push(`指标解析：${fmtYi(m.now)}（上期 ${fmtYi(m.prev)}，同比 ${m.yoy ?? "—"}%）`);
    }
  }
  out.push({
    id: q.id,
    question: q.question,
    answer:
      bullets.map((b) => `[${b.citation}] ${b.text}`).join("\n") + (extra.length ? "\n" + extra.join("\n") : ""),
  });
}

const path = root + "data/out/system_answers.json";
writeFileSync(path, JSON.stringify(out, null, 1), "utf8");
console.log("wrote", path);
for (const o of out) {
  console.log("=".repeat(90));
  console.log(o.id, o.question);
  console.log(o.answer);
}
