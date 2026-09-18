# -*- coding: utf-8 -*-
"""
make_dashboard.py — 把圖表＋63 筆清單打包成單一自帶式 HTML（圖片以 base64 內嵌）。

用法：python make_dashboard.py
產出：report_dashboard.html（在專案資料夾內，可直接雙擊開啟或丟到任何瀏覽器）
"""
import base64
import csv
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "report_assets")

CHARTS = [
    ("01_memory_bandwidth.png", "圖 1　記憶體階層與帶寬落差",
     "顯存 448 GB/s｜DDR4-2666 雙通道約 40 GB/s｜NVMe SSD 約 3 GB/s。權重每往下一層放，讀取代價就多一個量級。"),
    ("02_bandwidth_ceiling_vs_measured.png", "圖 2　帶寬上限 vs 實測（63 筆）",
     "每個點是一顆模型：x 軸＝每產生 1 個 token 需讀取的權重（MoE 只計活躍參數），y 軸＝實測 TKS。點離參考線的距離＝kernel 效率與 KV 開銷。"),
    ("03_size_vs_pool.png", "圖 3　模型載入容量 vs 30 GB 池",
     "綠＝純顯存（≤8 GB）；藍＝MoE 分層可行（8-30 GB）；橘＝dense 大模型（容量可以、速度鎖死）；灰＝依比例外推的估算值。"),
    ("04_results_distribution.png", "圖 4　63 筆執行結果分布",
     "41 筆能跑完、17 筆死循環／崩潰、5 筆無法執行。右圖：量化位元越低，死循環比例越高。"),
    ("05_ncmoe_tiering.png", "圖 5　MoE ＋ NCMoE 分層配置",
     "顯存放 attention／共享專家／熱門專家層；RAM 放其餘專家。路由決定「這個 token 要讀哪幾個專家」——這就是 8 GB 顯存能吃下 30B+ 的關鍵。"),
    ("06_capacity_decision.png", "圖 6　容量決策線",
     "先看塞不塞得下（8 GB / 30 GB 兩條線），再看是不是 MoE。容量過關但模型是 dense，速度依然不可用。"),
]

LIMITS = [
    ("① 池的硬上限 30 GB",
     "45B Q4／70B Q3 幾乎不留 KV/context 空間，再上去只能借 SSD。",
     "70B Q3 約 29.8 GB → 只剩約 0.2 GB 給 KV 與 context"),
    ("② 模型類型：dense 沒有專家可分",
     "dense 每個 token 都要讀全部權重，速度直接鎖在 RAM 帶寬以下。",
     "Muse-Glimmer-30B Q4 ＝ 1.5 TKS｜DeepSeek-R1-32B IQ2_M ＝ 2 TKS（容量其實塞得下）"),
    ("③ 格式／架構支援性",
     "不是「慢」而是「載不起來」或「跑起來是壞的」。",
     "TQ1_0/TQ2_0 需專用 fork｜K2-Horizon-7B 與 llama-b11002 CUDA 不相容｜amd.Instella-MoE PR 未合併｜ERNIE-4.5 輸出被導入 log｜125B UltraLite 輸出無效"),
]

POOL = [
    ("顯存 VRAM", "8 GB（實務 7.4～7.8）", "448 GB/s", "attention、KV cache、共享專家、熱門專家層"),
    ("系統 RAM", "約 22 GB", "約 35～40 GB/s", "其餘專家的權重（MoE）、被卸載的 dense 層"),
    ("合計", "約 30 GB", "—", "這就是「能載入多大模型」的依據"),
]

RECIPES = [
    ("純顯存小模型（≤12B Q4）", "-ngl 99 -c 8192 -fa --cache-type-k q8_0 --cache-type-v q8_0", "60～84 TKS；KV 用 q8_0 換回 context"),
    ("MoE 30B+（A3B/A4B）", "-ngl 99 -ncmoe 12 -c 4096 -fa --cache-type-k q4_0 --cache-type-v q4_0", "從 12 起掃；太小載不進去、太大掉到 10 TKS 以下"),
    ("要品質（26B-A4B Q2）", "-ngl 99 -ncmoe 20 -c 4096 -fa --draft-model <Q8MTP> --draft-max 1", "思考模式 10 TKS → 加 Q8MTP 投機解碼 35 TKS"),
    ("一直死循環", "--dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 --repeat-penalty 1.1", "先開 DRY，無效再降一級量化或關思考"),
]


def b64(name):
    with open(os.path.join(ASSETS, name), "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def load_models():
    """把中文欄位的 models.csv 轉成前端用的英文鍵（含結果正規化為 OK/LOOP/NA）。"""
    p = os.path.join(HERE, "models.csv")
    with open(p, encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    rmap = {"✅ 跑得完": "OK", "❌ 死循環/崩潰": "LOOP", "⚠️ 無法執行": "NA"}
    out = []
    for r in raw:
        res_raw = (r.get("執行結果") or "").strip()
        out.append({
            "model": (r.get("模型名稱 / 檔案名稱") or "").strip(),
            "result": rmap.get(res_raw, "NA"),
            "resultText": res_raw or "—",
            "tks": (r.get("推理速度 (TKS)") or "").strip(),
            "quality": (r.get("品質") or "").strip(),
            "eval": (r.get("測試評語摘要") or "").strip(),
            "note": f'{(r.get("來源表") or "").strip()}｜連結核對：{(r.get("連結核對狀態") or "").strip()}',
            "link": (r.get("連結") or "").strip(),
        })
    return out


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main():
    models = load_models()
    charts_html = "\n".join(
        f'''  <figure class="card chart">
    <img src="data:image/png;base64,{b64(fn)}" alt="{esc(title)}" loading="lazy">
    <figcaption><b>{esc(title)}</b><br>{esc(cap)}</figcaption>
  </figure>''' for fn, title, cap in CHARTS)

    pool_rows = "\n".join(
        f"<tr><td>{esc(a)}</td><td><b>{esc(b)}</b></td><td>{esc(c)}</td><td>{esc(d)}</td></tr>"
        for a, b, c, d in POOL)

    lim_html = "\n".join(
        f'''  <div class="limit"><h4>{esc(t)}</h4><p>{esc(mech)}</p>
    <p class="evidence">實測反例：{esc(ev)}</p></div>''' for t, mech, ev in LIMITS)

    recipe_rows = "\n".join(
        f"<tr><td>{esc(a)}</td><td><code>{esc(b)}</code></td><td>{esc(c)}</td></tr>"
        for a, b, c in RECIPES)

    payload = json.dumps(models, ensure_ascii=False)

    html = f'''<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3060 Ti 8G ＋ 32G DDR4-2666 本機 LLM 實測報告（圖表版）</title>
<style>
  :root {{ --bg:#0f1216; --card:#171b21; --line:#2a313b; --fg:#e8edf4; --dim:#9aa7b6;
           --ok:#4fd18b; --bad:#ff6b6b; --warn:#ffb454; --vram:#4fd18b; --ram:#5aa9ff; --ssd:#ff9f43; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font-family:"Segoe UI","Microsoft JhengHei",system-ui,sans-serif; line-height:1.65; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:28px 18px 60px; }}
  h1 {{ font-size:26px; margin:0 0 6px; }}
  h2 {{ font-size:19px; margin:38px 0 12px; padding-left:10px; border-left:4px solid var(--ram); }}
  h4 {{ margin:0 0 6px; font-size:15px; }}
  .sub {{ color:var(--dim); font-size:14px; margin-bottom:18px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 4px; }}
  .chip {{ background:var(--card); border:1px solid var(--line); border-radius:999px;
           padding:5px 12px; font-size:13px; color:var(--dim); }}
  .chip b {{ color:var(--fg); }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:12px; margin:16px 0 8px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; }}
  .kpi .n {{ font-size:26px; font-weight:700; }}
  .kpi .l {{ color:var(--dim); font-size:12.5px; }}
  .kpi.ok .n {{ color:var(--ok); }} .kpi.bad .n {{ color:var(--bad); }} .kpi.warn .n {{ color:var(--warn); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; }}
  .chart {{ margin:0 0 18px; padding:10px; }}
  .chart img {{ width:100%; height:auto; border-radius:8px; display:block; }}
  .chart figcaption {{ font-size:13px; color:var(--dim); padding:10px 6px 4px; }}
  .chart figcaption b {{ color:var(--fg); font-size:14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  th, td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
  th {{ color:var(--dim); font-weight:600; background:#141920; position:sticky; top:0; z-index:2; }}
  code {{ background:#0b0e12; border:1px solid var(--line); border-radius:5px; padding:1px 6px;
          font-size:12.5px; color:#c9e2ff; }}
  .limits {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; }}
  .limit {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--warn);
            border-radius:12px; padding:14px; }}
  .limit p {{ margin:6px 0 0; font-size:13.5px; color:var(--dim); }}
  .limit .evidence {{ color:#ffd9a8; }}
  .toolbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:12px 0; }}
  input[type=search] {{ background:#0b0e12; border:1px solid var(--line); color:var(--fg);
                        border-radius:8px; padding:8px 12px; font-size:14px; min-width:240px; }}
  button.f {{ background:var(--card); border:1px solid var(--line); color:var(--dim);
              border-radius:8px; padding:7px 12px; font-size:13px; cursor:pointer; }}
  button.f.on {{ color:#0f1216; background:var(--fg); border-color:var(--fg); font-weight:600; }}
  .mini {{ font-size:12.5px; color:var(--dim); }}
  .res {{ font-weight:600; white-space:nowrap; }}
  .res.OK {{ color:var(--ok); }} .res.LOOP {{ color:var(--bad); }} .res.NA {{ color:var(--dim); }}
  .scroll {{ max-height:620px; overflow:auto; border:1px solid var(--line); border-radius:12px; background:var(--card); }}
  .foot {{ color:var(--dim); font-size:12.5px; margin-top:34px; border-top:1px solid var(--line); padding-top:14px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>3060 Ti 8G ＋ 32G DDR4-2666　本機 LLM 實測報告（圖表版）</h1>
  <div class="sub">測試條件：Context 4096、Q4KV、單次任務「1,000 字推理小說」｜樣本 63 筆｜評分 GPT-SOL（6.0 及格）</div>
  <div class="chips">
    <span class="chip">GPU <b>RTX 3060 Ti 8 GB</b>｜<b>448 GB/s</b></span>
    <span class="chip">RAM <b>32 GB DDR4-2666</b>｜約 <b>40 GB/s</b></span>
    <span class="chip">可用池 <b>約 30 GB</b>（顯存 8 GB ＋ RAM 約 22 GB）</span>
    <span class="chip">工具 <b>GGUFRun</b></span>
  </div>

  <div class="kpis">
    <div class="kpi ok"><div class="n">41 / 63</div><div class="l">能跑完（65%）</div></div>
    <div class="kpi bad"><div class="n">17 / 63</div><div class="l">死循環或崩潰（27%）</div></div>
    <div class="kpi warn"><div class="n">5 / 63</div><div class="l">根本無法執行</div></div>
    <div class="kpi"><div class="n">11×</div><div class="l">顯存 vs RAM 帶寬落差</div></div>
    <div class="kpi"><div class="n">35B Q3</div><div class="l">MoE 分層實測可行上限（25 TKS）</div></div>
  </div>

  <h2>核心：可用池是 30 GB，但是兩個速度差 11 倍的層</h2>
  <table><thead><tr><th>層</th><th>可用容量</th><th>帶寬</th><th>放什麼</th></tr></thead>
  <tbody>{pool_rows}</tbody></table>
  <p class="mini">「塞得下」與「跑得動」是兩個獨立關卡：權重只要從顯存掉到 RAM，每讀 1 GB 的成本就多 11 倍。</p>

  <h2>六張圖看懂</h2>
{charts_html}

  <h2>破局：MoE ＋ NCMoE 分層，以及它的三個極限</h2>
  <div class="limits">
{lim_html}
  </div>

  <h2>GGUFRun 設定速查（3060 Ti 8G 專用）</h2>
  <table><thead><tr><th>情境</th><th>關鍵參數</th><th>說明</th></tr></thead>
  <tbody>{recipe_rows}</tbody></table>
  <p class="mini">專案內 <code>ggufrun_recipe.md</code> 有更完整的掃描流程與急救表。</p>

  <h2>63 筆模型瀏覽（搜尋／篩選）</h2>
  <div class="toolbar">
    <input type="search" id="q" placeholder="搜尋模型名稱、評語…">
    <button class="f on" data-f="all">全部</button>
    <button class="f" data-f="OK">能跑完</button>
    <button class="f" data-f="LOOP">死循環/崩潰</button>
    <button class="f" data-f="NA">無法執行</button>
    <span class="mini" id="cnt"></span>
  </div>
  <div class="scroll"><table><thead><tr><th>模型</th><th>結果</th><th>TKS</th><th>品質</th><th>評語</th><th>連結</th></tr></thead>
  <tbody id="rows"></tbody></table></div>

  <div class="foot">
    資料來源：作者實測（2026-09-18），本專案對 63 筆評語與 TKS 數值一字未改。<br>
    圖表由 <code>make_charts.py</code> 以同一份數據重繪；可稽核數據在 <code>chart_data.csv</code>。<br>
    工具：<a href="https://github.com/BBQ2077/GGUFRun" style="color:#5aa9ff">GGUFRun</a>（本機 GGUF 模型管理器）
  </div>
</div>

<script>
const MODELS = {payload};
const KEY = (o) => (o.result || "").toUpperCase();
const rowsEl = document.getElementById("rows"), qEl = document.getElementById("q"), cntEl = document.getElementById("cnt");
let filter = "all";
function norm(s) {{ return (s||"").trim(); }}
function render() {{
  const q = qEl.value.trim().toLowerCase();
  const out = MODELS.filter(m => {{
    if (filter !== "all" && KEY(m) !== filter) return false;
    if (!q) return true;
    return Object.values(m).join(" ").toLowerCase().includes(q);
  }});
  rowsEl.innerHTML = out.map(m => `<tr>
    <td><b>${{norm(m.model)}}</b><div class="mini">${{norm(m.note)}}</div></td>
    <td class="res ${{KEY(m)}}">${{norm(m.resultText)}}</td>
    <td>${{norm(m.tks) || "—"}}</td>
    <td>${{norm(m.quality) || "—"}}</td>
    <td>${{norm(m.eval)}}</td>
    <td>${{m.link && m.link.startsWith("http") ? `<a href="${{m.link}}" target="_blank" rel="noopener" style="color:#5aa9ff">HF</a>` : "—"}}</td></tr>`).join("");
  cntEl.textContent = `顯示 ${{out.length}} / ${{MODELS.length}} 筆`;
}}
document.querySelectorAll("button.f").forEach(b => b.addEventListener("click", () => {{
  document.querySelectorAll("button.f").forEach(x => x.classList.remove("on"));
  b.classList.add("on"); filter = b.dataset.f; render();
}}));
qEl.addEventListener("input", render);
render();
</script>
</body>
</html>'''
    outp = os.path.join(HERE, "report_dashboard.html")
    with open(outp, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print("寫入", outp, os.path.getsize(outp), "bytes")
    print("模型筆數:", len(models), "| 欄位:", list(models[0].keys()))


if __name__ == "__main__":
    main()
