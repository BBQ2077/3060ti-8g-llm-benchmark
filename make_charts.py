# -*- coding: utf-8 -*-
"""
make_charts.py — 產生 3060 Ti 8G + 32G DDR4-2666 報告的輔助圖表。

數據來源：
  * 實測評語/TKS  = 原始報告 llm_evaluation_4096_q4kv_report.md（人工逐筆登錄，不經正則分診）
  * 參數量/量化位元 = 由檔名推得，推不出者以 None 表示（該模型不進「大小相關」圖，避免編造）
  * 帶寬模型      = 每 token 需讀取的權重 約 (MoE ? active : total) 參數 × bits/8
                    天花板 t/s 約 可用帶寬 ÷ 每 token 讀取量
  * 這是「一階估算」：不計 KV、attention 開銷、kernel 效率、頁面交換抖動。
    用途是指出「速度牆在哪」，不是預測精確值。

輸出：report_assets/*.png + chart_data.csv
"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_assets")
os.makedirs(OUT, exist_ok=True)

font_manager.fontManager.addfont(r"C:\Windows\Fonts\msjh.ttc")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]  # 後者補 mathtext 的 U+2212 等字
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.edgecolor"] = "#bbb"

GREEN, RED, GRAY, BLUE, ORANGE = "#2e9e5b", "#d64545", "#9aa0a6", "#3d7dd8", "#e8912d"

# (顯示名, 總參數B, 活躍參數B(None=非MoE), 位元/參數, 實測TKS, 結果, 來源表)
D = [
 ("Gemma-4-E4B Uncensored Q4_K_P",        4.0,   None, 4.85, 77,   "OK",   "main"),
 ("Bonsai-27B-1bit-CRACK Q1_0",           27.0,  None, 1.50, None, "LOOP", "main"),
 ("Bonsai-27B Q1_0 (官方)",               27.0,  None, 1.50, None, "LOOP", "main"),
 ("Bonsai-27B-dspark-dflash Q4_0",        27.0,  None, 4.50, None, "OK",   "main"),
 ("MiniCPM5-2B Q4_K_M",                   2.0,   None, 4.85, 120,  "OK",   "main"),
 ("Spark-X2.5-4B Q4_K_M",                 4.0,   None, 4.85, 100,  "OK",   "main"),
 ("K2-Horizon-7B Q4_K_M",                 7.0,   None, 4.85, None, "NA",   "main"),
 ("Ornith-1.5-9B Q4_K_M",                 9.0,   None, 4.85, 60,   "OK",   "main"),
 ("Qwen3.8-4B-Distill Q4_K_M",            4.0,   None, 4.85, 100,  "OK",   "main"),
 ("maple-preview TQ1_0",                  None,  None, 1.60, 38,   "OK",   "main"),
 ("gemma4-v2 Q3_K_M",                     12.0,  None, 3.40, None, "LOOP", "main"),
 ("Huihui-Qwen3-8B-ablit v2 Q4_K_M",      8.0,   None, 4.85, 70,   "LOOP", "main"),
 ("Qwen3.8-27B RVN Heretic IQ1_S",        27.0,  None, 1.56, 30,   "LOOP", "main"),
 ("Qwen3.8-9B Q4_K_M",                    9.0,   None, 4.85, 67,   "OK",   "main"),
 ("Ling-3.0-tiny Q4_K_M",                 3.0,   None, 4.85, 158,  "LOOP", "main"),
 ("LFM2.5-2.6B Q4_K_M",                   2.6,   None, 4.85, 164,  "OK",   "main"),
 ("Phi-4-mini Q4_K_M",                    3.8,   None, 4.85, 116,  "LOOP", "main"),
 ("Qwen3.6-14B-A3B FableVibes Q2_K",      14.0,  3.0,  2.60, 85,   "LOOP", "main"),
 ("Qwen3-8B Q4_K_M (官方)",               8.0,   None, 4.85, 70,   "OK",   "main"),
 ("OxCoder-9B Q4_K_M",                    9.0,   None, 4.85, 67,   "OK",   "main"),
 ("Nanbeige4.2-3B Q4_K_M",                3.0,   None, 4.85, 60,   "LOOP", "main"),
 ("Gemma4-12B-QAT Uncensored",            12.0,  None, 4.85, 12,   "OK",   "main"),
 ("gemma-4-12b-it-UD Q2_K_XL",            12.0,  None, 2.60, 43,   "LOOP", "main"),
 ("glm-4-9b-chat Q4_K_S",                 9.0,   None, 4.60, 50,   "OK",   "main"),
 ("Mistral-7B-Instruct-v0.3 Q4_K_M",      7.0,   None, 4.85, 16,   "OK",   "main"),
 ("Qwen3.5-9B Defiant-Fable IQ3_M",       9.0,   None, 3.66, 24,   "OK",   "main"),
 ("maple TQ2_0",                          None,  None, 1.80, None, "NA",   "main"),
 ("Qwythos-9B Mythos MTP Q4_K_M",         9.0,   None, 4.85, 10,   "LOOP", "main"),
 ("Qwen3.8-27B-UD IQ1_S",                 27.0,  None, 1.56, 9,    "LOOP", "main"),
 ("DeepSeek-R1-0528-Qwen3-8B Q4_K_M",     8.0,   None, 4.85, 35,   "OK",   "main"),
 ("gemma-4-12B-it-qat-UD Q4_K_XL",        12.0,  None, 4.85, 13,   "OK",   "main"),
 ("gemma-4-E4B-it-qat-UD Q4_K_XL",        4.0,   None, 4.85, 65,   "OK",   "main"),
 ("Hermes-3-Llama-3.1-8B Q4_K_M",         8.0,   None, 4.85, 50,   "OK",   "main"),
 ("DeepSeek-V4-Pro-Qwen3.5-9B-MTP Q4_K_M",9.0,   None, 4.85, 30,   "LOOP", "main"),
 ("DeepSeek-Coder-V2-Lite IQ2_M",         16.0,  2.4,  2.70, 70,   "OK",   "main"),
 ("DeepSeek-R1-Distill-Qwen-14B IQ2_M",   14.0,  None, 2.70, 25,   "LOOP", "main"),
 ("DeepSeek-R1-Distill-Qwen-32B IQ2_M",   32.0,  None, 2.70, 2,    "OK",   "main"),
 ("internlm3-8b-instruct q4_k_m",         8.0,   None, 4.85, 67,   "OK",   "main"),
 ("Yi-1.5-9B-Chat Q4_K_M",                9.0,   None, 4.85, 67,   "OK",   "main"),
 ("aya-expanse-8b Q4_K_M",                8.0,   None, 4.85, 70,   "OK",   "main"),
 ("Qwen2.5-7B-Instruct Q4_K_M",           7.0,   None, 4.85, None, "OK",   "main"),
 ("Qwen3-14B-Claude-4.5-Opus-Distill Q3_K_S", 14.0, None, 3.30, 25, "OK",  "main"),
 ("TieFighter-Holodeck-20B IQ2_M",        20.0,  None, 2.70, 14,   "OK",   "main"),
 ("Moonlight-16B-A3B IQ2_XS",             16.0,  3.0,  2.50, None, "OK",   "main"),
 ("amd.Instella-MoE-16B-A3B Q2_K",        16.0,  3.0,  2.60, None, "NA",   "main"),
 ("bohf-12b-moe-3a q4_k_m",               12.0,  3.0,  4.85, None, "LOOP", "main"),
 ("gemma-4-E4B-it-OBLITERATED i1-Q4_K_S", 4.0,   None, 4.60, 84,   "OK",   "main"),
 ("gemma-4-26B-A4B-it-UD Q2_K_XL",        26.0,  4.0,  2.60, 10,   "OK",   "main"),
 ("gemma-4-26B-A4B + Q8MTP (n_max=1)",    26.0,  4.0,  2.60, 35,   "OK",   "moe"),
 ("Qwen3.6-35B-A3B-UD Q3_K_XL",           35.0,  3.0,  3.40, 25,   "OK",   "moe"),
 ("gpt-oss-20b Q8_0",                     20.0,  3.6,  8.50, 22,   "OK",   "moe"),
 ("ERNIE-4.5-21B-A3B-Thinking Q4_K_XL",   21.0,  3.0,  4.85, None, "NA",   "moe"),
 ("Nemotron-3-Nano 30B-A3B Q4_0",         30.0,  3.0,  4.50, 12,   "OK",   "moe"),
 ("GLM-4.7-Flash-UD Q3_K_XL",             None,  None, 3.40, 30,   "OK",   "moe"),
 ("Ornith-1.5-35B-A3B IQ4_XS",            35.0,  3.0,  4.25, 37,   "OK",   "moe"),
 ("Ternary-Bonsai-2-27B PTQ1_0",          27.0,  None, 1.60, 28,   "OK",   "moe"),
 ("Qwen3.8-Flash-Next-UD Q2_K_XL (reap320)", None, None, 2.60, 10, "LOOP", "moe"),
 ("K2-Horizon-MoVA-36B-A4B Q3_K_M",       36.0,  4.0,  3.40, 13,   "LOOP", "moe"),
 ("Qwen3.8-Flash-Next-125B UltraLite-37GiB", 125.0, 6.0, 2.37, 6, "OK",   "moe"),
 ("Nex-N2.5-mini IQ2_M",                  None,  None, 2.70, 39,   "OK",   "moe"),
 ("Huihui4-48B-A4B IQ3_M",                48.0,  4.0,  3.66, 17,   "OK",   "moe"),
 ("JoyAI-LLM-Flash IQ3_XS",               None,  None, 3.30, 21,   "OK",   "moe"),
 ("Muse-Glimmer-30B-UD Q4_K_XL",          30.0,  None, 4.85, 1.5,  "OK",   "moe"),
]

VRAM_BW, RAM_BW = 448.0, 40.0      # GB/s（3060 Ti GDDR6 448；DDR4-2666 雙通道理論 42.7、實務約 35～40）
SSD_BW = 3.0                       # NVMe 樂觀值；SATA 約 0.55

def per_token_gb(total, active, bits):
    p = active if active else total
    return None if p is None else p * bits / 8.0

def size_gb(total, bits):
    return None if total is None else total * bits / 8.0

# ---------- 稽核用數據表 ----------
with open(os.path.join(HERE, "chart_data.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["模型", "總參數B", "活躍參數B", "位元/參數", "估算每token讀取GB", "估算載入大小GB",
                "實測TKS", "執行結果", "帶寬牆(VRAM/RAM/SSD)", "估算上限TKS"])
    for name, tot, act, bits, tks, res, tbl in D:
        pt, sz = per_token_gb(tot, act, bits), size_gb(tot, bits)
        if pt is None:
            wall, ceil = "無法估算", ""
        elif sz is not None and sz <= 8.0:
            wall, ceil = "VRAM", round(VRAM_BW / pt, 1)
        elif sz is not None and sz <= 30.0:
            wall, ceil = "RAM", round(RAM_BW / pt, 1)
        else:
            wall, ceil = "SSD/超池", round(SSD_BW / pt, 1)
        w.writerow([name, tot, act, bits, round(pt, 3) if pt else "", round(sz, 1) if sz else "",
                    tks if tks else "", res, wall, ceil])

# ---------- 圖 1：記憶體階層與帶寬落差 ----------
fig, ax = plt.subplots(figsize=(8.4, 3.2))
labels = ["顯存 VRAM\n8 GB", "系統 RAM\n22 GB（可用）", "NVMe SSD\n（分層救援）"]
vals   = [VRAM_BW, RAM_BW, SSD_BW]
cols   = [GREEN, BLUE, ORANGE]
b = ax.barh(range(3), vals, color=cols, height=0.55)
ax.set_xscale("log")
for i, v in enumerate(vals):
    ax.text(v * 1.15, i, f"約 {v:g} GB/s", va="center", fontsize=10,
            color="#222", fontweight="bold")
ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("可用記憶體帶寬（GB/s，對數刻度）", fontsize=9)
ax.set_title("速度的牆：顯存比 RAM 快約 11 倍、RAM 比 SSD 快約 13 倍（顯存 vs SSD 約 150 倍）\n"
             "模型每產生 1 個 token，都要把「這個 token 會用到的權重」重讀一遍（x 軸為對數刻度）", fontsize=9.8)
ax.annotate("", xy=(VRAM_BW, 2.42), xytext=(RAM_BW, 2.42),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.1))
ax.text((VRAM_BW * RAM_BW) ** 0.5, 2.5, "11 倍", ha="center", fontsize=9, color="#555")
ax.annotate("", xy=(RAM_BW, 2.42), xytext=(SSD_BW, 2.42),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.1))
ax.text((RAM_BW * SSD_BW) ** 0.5, 2.5, "13 倍", ha="center", fontsize=9, color="#555")
ax.set_ylim(-0.6, 2.9); ax.grid(axis="x", alpha=.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "01_memory_bandwidth.png")); plt.close()

# ---------- 圖 2：帶寬天花板 vs 實測散點 ----------
fig, ax = plt.subplots(figsize=(9.4, 5.8))
grid = [0.05 * (1.35 ** i) for i in range(44)]
for bw, c, lb in ((VRAM_BW, GREEN, "顯存帶寬上限 448 GB/s"),
                  (RAM_BW, BLUE, "DDR4-2666 雙通道上限 約 40 GB/s"),
                  (SSD_BW, ORANGE, "SSD 分層上限 約 3 GB/s")):
    ax.plot(grid, [bw / g for g in grid], "--", color=c, lw=1.4, alpha=.85, label=lb, zorder=2)
for res, c, lb in (("OK", GREEN, "實測：能跑完"), ("LOOP", RED, "實測：死循環/崩潰")):
    pts = [(per_token_gb(t, a, b), tk) for n, t, a, b, tk, r, _ in D
           if r == res and tk and per_token_gb(t, a, b)]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=c, s=58, edgecolor="white",
               linewidth=.8, zorder=5, label=lb)
# 只標註具代表性的點，位置人工錯開（避免標籤互壓）
NOTE = {"LFM2.5-2.6B Q4_K_M": (9, 5), "Hermes-3-Llama-3.1-8B Q4_K_M": (-40, 20),
        "gemma-4-26B-A4B + Q8MTP (n_max=1)": (15, 9), "Qwen3.6-35B-A3B-UD Q3_K_XL": (15, -15),
        "Huihui4-48B-A4B IQ3_M": (-48, 13), "gpt-oss-20b Q8_0": (13, 11),
        "DeepSeek-R1-Distill-Qwen-32B IQ2_M": (15, 5), "Muse-Glimmer-30B-UD Q4_K_XL": (15, -5)}
for name, tot, act, bits, tks, res, tbl in D:
    pt = per_token_gb(tot, act, bits)
    if name in NOTE and pt and tks:
        ax.annotate(name.split(" (")[0][:24], (pt, tks), fontsize=6.8, color="#3a3a3a",
                    xytext=NOTE[name], textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=.6))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.03, 70); ax.set_ylim(0.7, 320)
ax.set_xlabel("每 token 需讀取的權重（GB，估算；MoE 只計活躍參數）", fontsize=9)
ax.set_ylabel("實測速度（TKS，對數刻度）", fontsize=9)
ax.set_title("實測點落在哪條牆下：點離參考線的距離 = kernel 效率 + KV/attention 開銷\n"
             "左上角（小而快）＝全在顯存；右下角（大而慢）＝權重被擠到 RAM", fontsize=10.2)
ax.legend(fontsize=8, loc="lower left", framealpha=.92)
ax.grid(alpha=.25, which="both")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "02_bandwidth_ceiling_vs_measured.png")); plt.close()

# ---------- 圖 3：模型載入大小 vs 30 GB 可用池 ----------
sel = [
 ("Qwen3.8-4B-Distill Q4", 4.0, 4.85, 100, "dense"),
 ("Hermes-3-Llama-3.1-8B Q4", 8.0, 4.85, 50, "dense"),
 ("gemma-4-26B-A4B Q2", 26.0, 2.60, 10, "moe"),
 ("Qwen3.6-35B-A3B Q3", 35.0, 3.40, 25, "moe"),
 ("gpt-oss-20b Q8", 20.0, 8.50, 22, "moe"),
 ("Huihui4-48B-A4B IQ3_M", 48.0, 3.66, 17, "moe"),
 ("Muse-Glimmer-30B Q4 (dense)", 30.0, 4.85, 1.5, "dense"),
 ("DeepSeek-R1-32B IQ2_M (dense)", 32.0, 2.70, 2, "dense"),
 ("45B Q4（估算）", 45.0, 4.85, None, "est"),
 ("70B Q3（估算）", 70.0, 3.40, None, "est"),
 ("90B Q2（估算）", 90.0, 2.60, None, "est"),
 ("100B IQ1（估算）", 100.0, 1.70, None, "est"),
]
fig, ax = plt.subplots(figsize=(8.8, 4.6))
names = [s[0] for s in sel][::-1]
sizes = [s[1] * s[2] / 8 for s in sel][::-1]
tks   = [s[3] for s in sel][::-1]
kind  = [s[4] for s in sel][::-1]
def col_of(k, sz):
    if k == "est":   return "#b9b9b9"
    if sz <= 8.0:    return GREEN
    if k == "moe":   return BLUE
    return ORANGE
cols = [col_of(k, sz) for k, sz in zip(kind, sizes)]
ax.barh(range(len(sel)), sizes, color=cols, height=.62)
ax.axvline(8, color=GREEN, ls="--", lw=1.4)
ax.axvline(30, color=RED, ls="--", lw=1.4)
ax.text(8, len(sel) - .2, " 8 GB 顯存", color=GREEN, fontsize=8.5, va="bottom")
ax.text(30, len(sel) - .2, " 30 GB 池（8 顯存 + 22 RAM）", color=RED, fontsize=8.5, va="bottom")
for i, (nm, sz, t) in enumerate(zip(names, sizes, tks)):
    lab = f"{sz:.1f} GB"
    if t: lab += f"  → 實測 {t:g} TKS"
    if sz >= 26:   # 靠線的標籤改放進長條內側，避免被 30GB 虛線切過
        ax.text(sz - 0.7, i, lab, va="center", ha="right", fontsize=7.8, color="white",
                fontweight="bold")
    else:
        ax.text(sz + 0.6, i, lab, va="center", fontsize=7.8, color="#333")
ax.set_yticks(range(len(sel))); ax.set_yticklabels(names, fontsize=8.6)
ax.set_xlabel("模型載入所需容量（GB，估算）", fontsize=9)
ax.set_xlim(0, 48)
ax.set_title("塞得下 ≠ 跑得動：容量過關的，速度未必過關\n"
             "綠＝純顯存；藍＝MoE 分層可行；橘＝dense 大模型（容量可以、速度鎖死）；灰＝外推估算", fontsize=10)
ax.grid(axis="x", alpha=.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "03_size_vs_pool.png")); plt.close()

# ---------- 圖 4：63 筆結果分布 ----------
ok   = sum(1 for d in D if d[5] == "OK")
loop = sum(1 for d in D if d[5] == "LOOP")
na   = sum(1 for d in D if d[5] == "NA")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.4), gridspec_kw={"width_ratios": [1, 1.15]})
a1.pie([ok, loop, na], labels=[f"能跑完\n{ok} 筆", f"死循環/崩潰\n{loop} 筆", f"無法執行\n{na} 筆"],
       colors=[GREEN, RED, GRAY], autopct="%1.0f%%", startangle=90,
       textprops={"fontsize": 9}, wedgeprops={"edgecolor": "white", "linewidth": 1.4})
a1.set_title("63 筆實測的執行結果", fontsize=10)
# 量化級別 → 死循環率
buckets = [("Q1/IQ1\n(<=2 bpw)", lambda b: b <= 2.0), ("Q2/IQ2\n(2-3 bpw)", lambda b: 2.0 < b <= 3.0),
           ("Q3\n(3-4 bpw)", lambda b: 3.0 < b <= 4.0), ("Q4+\n(>4 bpw)", lambda b: b > 4.0)]
bl, bc, btot = [], [], []
for lb, f in buckets:
    sub = [d for d in D if f(d[3])]
    bl.append(f"{lb}\n(n={len(sub)})")
    bc.append(sum(1 for d in sub if d[5] == "LOOP"))
    btot.append(len(sub))
a2.bar(bl, btot, color="#dfe4ea", label="測試筆數")
a2.bar(bl, bc, color=RED, label="死循環/崩潰")
for i, (c, t) in enumerate(zip(bc, btot)):
    if t: a2.text(i, t + .5, f"{c}/{t}", ha="center", fontsize=8.5, color="#333")
a2.set_ylabel("筆數", fontsize=9); a2.grid(axis="y", alpha=.25)
a2.set_title("量化位元越低，死循環比例越高", fontsize=10)
a2.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "04_results_distribution.png")); plt.close()

# ---------- 圖 5：NCMoE 分層配置示意 ----------
fig, ax = plt.subplots(figsize=(8.8, 4.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
def box(x, y, w, h, fc, txt, fs=8.4, tc="#123"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec="#888", lw=1))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc)
ax.text(5, 5.66, "MoE ＋ NCMoE：把「順序不需要一起讀」的專家拆去 RAM", ha="center",
        fontsize=11, fontweight="bold")
box(0.3, 3.5, 4.3, 1.75, "#dff3e4", "顯存 8 GB（448 GB/s）\n─────────────\nattention／KV／共享專家\n＋ 常被路由到的前幾層專家\n（-ncmoe N：前 N 層專家留 CPU）")
box(5.4, 3.5, 4.3, 1.75, "#dfe9fb", "系統 RAM 22 GB（約40 GB/s）\n─────────────\n其餘專家的權重\n（每 token 只讀被路由到的那幾個 → 讀取量 約 3～4B 參數）")
box(0.3, 1.5, 4.3, 1.45, "#fdf1e0", "SSD（約0.5-3 GB/s）\n─────────────\n超池救援：>30 GB 的模型\n→ 實測 1.5～6 TKS，抖動大")
box(5.4, 1.5, 4.3, 1.45, "#fbe3e3", "dense 大模型（無專家可分）\n─────────────\n每個 token 全部權重都要讀\n→ 30B Q4 實測 1.5 TKS、32B IQ2 2 TKS")
ax.add_patch(FancyArrowPatch((4.7, 4.35), (5.35, 4.35), arrowstyle="->", mutation_scale=14, color="#3d7dd8"))
ax.add_patch(FancyArrowPatch((5.35, 3.9), (4.7, 3.9), arrowstyle="->", mutation_scale=14, color="#3d7dd8"))
ax.text(5.02, 4.47, "路由（挑專家）", ha="center", fontsize=7.8, color="#3d7dd8")
ax.text(5.02, 3.72, "權重載入", ha="center", fontsize=7.8, color="#3d7dd8")
ax.text(0.3, 0.75, "實測證據：A3B/A4B 型（35B Q3 = 25 TKS、48B IQ3_M = 17 TKS、20B Q8 = 22 TKS）分層後仍可用；\n"
                   "dense 或「活躍參數太大」的模型，把權重放去 RAM 等於直接鎖在 40 GB/s 以下。（TKS = tokens/s，每秒生成 token 數）",
        fontsize=8.6, color="#333")
ax.text(0.3, 0.18, "支援性極限：TQ1_0/TQ2_0 需專用 fork、K2-Horizon 與 b11002 不相容、Instella-MoE 的 PR 尚未合併、"
                   "ERNIE-4.5 輸出被導入 log。", fontsize=8.2, color="#b03a3a")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "05_ncmoe_tiering.png")); plt.close()

# ---------- 圖 6：量化與「載入池」決策圖 ----------
fig, ax = plt.subplots(figsize=(8.8, 4.2))
cases = [("4B Q4", 4.0, 4.85), ("8B Q4", 8.0, 4.85), ("12B Q4", 12.0, 4.85),
         ("26B-A4B Q2", 26.0, 2.60), ("35B-A3B Q3", 35.0, 3.40), ("48B-A4B IQ3", 48.0, 3.66),
         ("70B Q3", 70.0, 3.40), ("90B Q2", 90.0, 2.60), ("125B IQ2", 125.0, 2.37)]
lbl = [c[0] for c in cases]; sz = [c[1] * c[2] / 8 for c in cases]
x = range(len(cases))
ax.bar(x, sz, color=[GREEN if s <= 8 else (BLUE if s <= 30 else ORANGE) for s in sz], width=.6)
ax.axhline(8, color=GREEN, ls=":", lw=1.2); ax.axhline(30, color=RED, ls=":", lw=1.2)
ax.text(len(cases) - .4, 8.6, "8 GB 顯存上限", color=GREEN, fontsize=8, ha="right")
ax.text(len(cases) - .4, 30.8, "30 GB 可用池上限（8 顯存 + 22 RAM）", color=RED, fontsize=8, ha="right")
for i, s in enumerate(sz):
    ax.text(i, s + .6, f"{s:.1f}", ha="center", fontsize=7.6, color="#333")
ax.set_xticks(list(x)); ax.set_xticklabels(lbl, fontsize=8.4)
ax.set_xlabel("模型規模與量化方式（A3B/A4B = MoE 活躍 3B/4B；IQ = i-quant 混合量化）", fontsize=8.6)
ax.set_ylabel("載入所需容量（GB，估算）", fontsize=9); ax.set_ylim(0, 42)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=GREEN, label="<=8 GB：純顯存可跑"),
                   Patch(facecolor=BLUE, label="8-30 GB：MoE 分層可跑（dense 大模型不適用）"),
                   Patch(facecolor=ORANGE, label=">30 GB：需 SSD 分層，實用性低")],
          fontsize=8, loc="upper left", framealpha=.95)
ax.set_title("容量決策線：先看塞不塞得下，再看是不是 MoE（dense 大模型即使塞得下也慢到不可用）", fontsize=9.8)
ax.grid(axis="y", alpha=.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "06_capacity_decision.png")); plt.close()

print("完成。產出：")
for f in sorted(os.listdir(OUT)):
    print("  report_assets/" + f, os.path.getsize(os.path.join(OUT, f)), "bytes")
print("  chart_data.csv", os.path.getsize(os.path.join(HERE, "chart_data.csv")), "bytes")
