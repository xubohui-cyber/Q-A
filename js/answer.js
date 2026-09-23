/* 答案组装：把检索到的原文块变成「要点 + 出处标记」。
 * 不依赖 onnxruntime，便于在 Node 里对同一份逻辑做离线评测。
 */

import { tokenize } from "./bm25.js";

export function queryTerms(query) {
  return new Set(tokenize(query).filter((t) => t.length > 1 || !/[a-z0-9]/.test(t)));
}

/** 从块里挑出与问题最相关的行（抽取式答案的要点来源） */
export function keyLines(text, terms, limit = 3) {
  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("| ---") && !/^\|?\s*-{2,}/.test(l));
  const scored = lines.map((l) => {
    const lt = new Set(tokenize(l));
    let hit = 0;
    for (const t of terms) if (lt.has(t)) hit += 1;
    return { line: l, hit, len: l.length };
  });
  scored.sort((a, b) => b.hit * 1000 + Math.min(b.len, 300) - (a.hit * 1000 + Math.min(a.len, 300)));
  return scored.filter((s) => s.hit > 0).slice(0, limit).map((s) => s.line);
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

export function markdownTableToHtml(md) {
  const rows = md.split("\n").filter((l) => l.trim().startsWith("|"));
  if (rows.length < 2) return null;
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const head = cells(rows[0]);
  const body = rows.slice(2).map(cells);
  return `<table class="md-table"><thead><tr>${head
    .map((h) => `<th>${escapeHtml(h)}</th>`)
    .join("")}</tr></thead><tbody>${body
    .map((r) => `<tr>${r.map((c) => `<td>${escapeHtml(c)}</td>`).join("")}</tr>`)
    .join("")}</tbody></table>`;
}

export function highlight(text, terms) {
  const esc = escapeHtml(text);
  const list = [...terms].filter((t) => t.length > 1).sort((a, b) => b.length - a.length).slice(0, 40);
  let out = esc;
  for (const t of list) {
    const safe = t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    out = out.replace(new RegExp(safe, "g"), (m) => `<mark>${m}</mark>`);
  }
  return out;
}

/** 抽取式答案：直接从检索到的原文里挑要点，并标注出处编号 */
export function composeExtractive(query, results, limit = 3) {
  const terms = queryTerms(query);
  const bullets = [];
  for (const c of results.slice(0, limit)) {
    const lines = keyLines(c.text, terms, 2);
    if (!lines.length && c.type === "table") {
      const rows = c.text.split("\n").filter((l) => l.trim().startsWith("|"));
      if (rows.length) lines.push(rows.slice(0, 2).join(" "));
    }
    if (!lines.length) lines.push(c.text.split("\n")[0]);
    bullets.push({ citation: c.rank, chunk: c, text: lines.join("；").slice(0, 220) });
  }
  return bullets;
}

/* ---------- 全景题：从「主要会计数据」表里抽指标 ---------- */

const NUM = /^-?[\d,]+(?:\.\d+)?%?$/;
const clean = (s) => s.replace(/[ 　]/g, "");

function toNum(cell) {
  const c = clean(cell);
  if (!NUM.test(c)) return null;
  const v = Number(c.replace(/[,%]/g, ""));
  return Number.isFinite(v) ? v : null;
}

function mergeSplit(cells) {
  const out = [];
  for (let i = 0; i < cells.length; i++) {
    const cur = clean(cells[i]);
    if (i + 1 < cells.length && /^-?[\d,]+\.$/.test(cur) && /^\d{1,2}$/.test(clean(cells[i + 1]))) {
      out.push(cur + clean(cells[i + 1]));
      i += 1;
    } else {
      out.push(cells[i]);
    }
  }
  return out;
}

function tables(text) {
  const out = [];
  let cur = [];
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (s.startsWith("|")) {
      if (/^[|\-:\s]+$/.test(s)) continue;
      cur.push(mergeSplit(s.replace(/^\||\|$/g, "").split("|").map((c) => c.trim())));
    } else if (cur.length) {
      out.push(cur);
      cur = [];
    }
  }
  if (cur.length) out.push(cur);
  return out;
}

function labelMatch(label, nextLabel, metric) {
  if (metric === "revenue") {
    if (!/营业收入|营业总收入/.test(label)) return false;
    return !/成本|构成|比重|占比|明细|分产品/.test(label);
  }
  if (!label.includes("归属于上市公司股东")) return false;
  if (label.includes("扣除非经常性损益")) return false;
  if (label.includes("净利润")) return true;
  return nextLabel.includes("净利润") && !nextLabel.includes("扣除非经常性损益");
}

/** 返回 {now, prev, yoy}（单位：元），找不到返回 null */
export function extractMetric(text, metric) {
  const unit = /单位\s*[:：]\s*(人民币)?万元/.test(text) ? 1e4 : 1;
  const found = [];
  for (const rows of tables(text)) {
    const header = rows.slice(0, 3).flat().map(clean).join("");
    if (header.includes("季度")) continue; // 分季度表是单季数据，不能当全年/半年值
    const score = (header.includes("主要会计数据") ? 2 : 0) + (header.includes("上年同期") ? 1 : 0);
    for (let i = 0; i < rows.length; i++) {
      const cells = rows[i];
      if (cells.length < 3) continue;
      const label = cells.map(clean).join("");
      const nextLabel = i + 1 < rows.length ? rows[i + 1].map(clean).join("") : "";
      if (!labelMatch(label, nextLabel, metric)) continue;
      const values = [];
      for (const c of cells) {
        const v = toNum(c);
        if (v !== null) values.push([v, c.trim().endsWith("%")]);
      }
      // 金额与同比按数值量级区分：全年/半年行是「本期 上期 (同比) 上年」，分季度行是 4 个季度金额
      const amounts = values.filter(([v]) => Math.abs(v) > 1000).map(([v]) => v);
      const smalls = values.filter(([v]) => Math.abs(v) <= 1000).map(([v]) => v);
      if (amounts.length < 2 || amounts.length > 3) continue;
      const now = amounts[0];
      const prev = amounts[1];
      let yoy = values.find(([v, p]) => p && Math.abs(v) <= 10000)?.[0] ?? null;
      if (yoy === null) {
        yoy = smalls.find((v) => Math.abs(v) !== Math.abs(now)) ?? null;
      }
      found.push({ score, now: now * unit, prev: prev * unit, yoy });
    }
  }
  if (!found.length) return null;
  return found.reduce((a, b) => {
    if (b.score !== a.score) return b.score > a.score ? b : a;
    return Math.abs(b.now) > Math.abs(a.now) ? b : a;
  });
}

export function fmtYi(v) {
  return v === null || v === undefined ? "—" : `${(v / 1e8).toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} 亿元`;
}
