/* 校验浏览器端 tokenizer.js 与 Python tokenizers 输出逐 token 一致 */
import { readFileSync } from "node:fs";
import { BertTokenizer } from "../js/tokenizer.js";

const root = new URL("../", import.meta.url).pathname;
const ref = JSON.parse(readFileSync(root + "data/out/tokenizer_ref.json", "utf8"));
const tkJson = JSON.parse(readFileSync(root + "model/tokenizer.json", "utf8"));
const tok = new BertTokenizer(tkJson);

let bad = 0;
for (const item of ref) {
  const { ids } = tok.encode(item.text);
  const same = ids.length === item.ids.length && ids.every((v, i) => v === item.ids[i]);
  if (!same) {
    bad++;
    console.log("MISMATCH:", JSON.stringify(item.text));
    console.log("  py:", item.ids.slice(0, 24).join(","));
    console.log("  js:", ids.slice(0, 24).join(","));
  }
}
console.log(bad === 0 ? `全部一致（${ref.length} 条）` : `${bad}/${ref.length} 条不一致`);
