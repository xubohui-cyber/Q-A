/* 用无头 Chrome 真机验证网页：索引加载、BM25 检索、全景模式、评测页渲染。
 * 运行：NODE_PATH=<bundled node_modules> node browser_test.cjs
 */
const { chromium } = require("playwright");

const BASE = process.env.BASE_URL || "http://127.0.0.1:8777/";
const SHOT = process.env.SHOT_DIR || "/tmp";

async function main() {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
  const errors = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`console: ${m.text()}`);
  });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

  console.log("→ 打开页面");
  await page.goto(BASE, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => document.querySelector("#s-chunks").textContent !== "–",
    null,
    { timeout: 180000 }
  );
  const stats = await page.evaluate(() => ({
    companies: document.querySelector("#s-companies").textContent,
    reports: document.querySelector("#s-reports").textContent,
    pages: document.querySelector("#s-pages").textContent,
    chunks: document.querySelector("#s-chunks").textContent,
    tables: document.querySelector("#s-tables").textContent,
    status: document.querySelector("#status").textContent.trim(),
  }));
  console.log("   统计:", JSON.stringify(stats));

  // 1) BM25 单公司提问
  console.log("→ Q1 贵州茅台营业收入（BM25 模式）");
  await page.fill("#q", "贵州茅台2025年营业收入和归母净利润是多少？");
  await page.click("#go");
  await page.waitForSelector(".ev", { timeout: 60000 });
  const q1 = await page.evaluate(() => ({
    bullets: [...document.querySelectorAll(".ans li")].map((li) => li.textContent.trim().slice(0, 160)),
    first: document.querySelector(".ev .evhead").textContent.replace(/\s+/g, " ").trim(),
    status: document.querySelector("#status").textContent.trim(),
    evCount: document.querySelectorAll(".ev").length,
  }));
  console.log("   状态:", q1.status);
  console.log("   证据数:", q1.evCount, "| #1:", q1.first);
  q1.bullets.forEach((b, i) => console.log(`   要点${i + 1}: ${b}`));
  await page.screenshot({ path: `${SHOT}/rag_q1.png`, fullPage: false });

  // 2) 全景模式
  console.log("→ Q8 全景模式（16 家公司对比）");
  await page.fill("#q", "全景：2025年营业收入最高的5家白酒公司是谁？");
  await page.check("#panorama");
  await page.click("#go");
  await page.waitForSelector(".panotable tbody tr", { timeout: 60000 });
  const pano = await page.evaluate(() => ({
    rows: [...document.querySelectorAll(".panotable tbody tr")].map((tr) =>
      [...tr.querySelectorAll("td")].map((td) => td.textContent.replace(/\s+/g, " ").trim()).slice(0, 3)
    ),
    status: document.querySelector("#status").textContent.trim(),
  }));
  console.log("   状态:", pano.status);
  pano.rows.slice(0, 6).forEach((r) => console.log("   ", r.join(" | ")));
  console.log("   共", pano.rows.length, "行");
  await page.screenshot({ path: `${SHOT}/rag_panorama.png`, fullPage: false });
  await page.uncheck("#panorama");

  // 3) 评测页
  console.log("→ 评测页渲染");
  await page.click('button[data-tab="eval"]');
  await page.waitForSelector(".evalitem", { timeout: 30000 });
  const evalInfo = await page.evaluate(() => ({
    items: document.querySelectorAll(".evalitem").length,
    summary: document.querySelector("#evalSummary").textContent.replace(/\s+/g, " ").trim(),
  }));
  console.log("   ", evalInfo.items, "题 |", evalInfo.summary);
  await page.screenshot({ path: `${SHOT}/rag_eval.png`, fullPage: false });

  // 4) 语义检索（下载 onnxruntime wasm + int8 模型）
  console.log("→ 启用语义检索（浏览器内跑 bge-small-zh-v1.5）");
  await page.click('button[data-tab="qa"]');
  await page.click("#enableDense");
  await page.waitForFunction(
    () => /语义检索已启用|失败/.test(document.querySelector("#status").textContent),
    null,
    { timeout: 300000 }
  );
  const denseStatus = await page.evaluate(() => document.querySelector("#status").textContent.trim());
  console.log("   状态:", denseStatus);

  if (/已启用/.test(denseStatus)) {
    await page.selectOption("#mode", "hybrid");
    await page.fill("#q", "五粮液2025年酒类产品的毛利率是多少？");
    await page.click("#go");
    await page.waitForSelector(".ev", { timeout: 60000 });
    const hy = await page.evaluate(() => ({
      status: document.querySelector("#status").textContent.trim(),
      bullets: [...document.querySelectorAll(".ans li")].map((li) => li.textContent.trim().slice(0, 150)),
    }));
    console.log("   混合检索:", hy.status);
    hy.bullets.forEach((b, i) => console.log(`   要点${i + 1}: ${b}`));
    await page.screenshot({ path: `${SHOT}/rag_hybrid.png`, fullPage: false });
  }

  console.log(errors.length ? `\n控制台错误 ${errors.length} 条:` : "\n无控制台错误");
  errors.slice(0, 10).forEach((e) => console.log("   ", e));
  await browser.close();
}

main().catch((e) => {
  console.error("测试失败:", e);
  process.exit(1);
});
