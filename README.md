# 3060 Ti 8G ＋ 32GB DDR4-2666 本機 LLM 實測報告

> **一句話**：要在 8 GB 顯存上跑得動、跑得快，關鍵不是模型多大，而是**每產生一個 token 要讀多少權重、從哪裡讀**。
> 這份專案用 **63 顆模型**的實測數據，把這台機器的「可行區間」畫出來，並給出可以直接照抄的參數與流程。
>
> 測試條件：**Context 4096、Q4KV**、單次任務「**撰寫 1,000 字推理小說**」｜評分：GPT-SOL（6.0 / 10 及格）

![GPU](https://img.shields.io/badge/GPU-RTX%203060%20Ti%208GB-76b900)
![RAM](https://img.shields.io/badge/RAM-32GB%20DDR4--2666-blue)
![Samples](https://img.shields.io/badge/樣本-63%20runs-orange)
![Tool](https://img.shields.io/badge/tool-GGUFRun%20v2.0-informational)

---

## 中文

### 專案內容

| 檔案 | 內容 |
| :--- | :--- |
| [`3060ti_8g_model_report.md`](3060ti_8g_model_report.md) | **主報告**。硬體基線、帶寬一致性檢查、**NCMoE 分層與三個極限**、63 筆實測總表、死循環六對策、決策樹、GGUFRun 操作章節、修正紀錄 |
| [`report_dashboard.html`](report_dashboard.html) | **一頁式儀表板**（自帶圖表，雙擊即可開）：KPI、六張圖、三個極限、設定速查、63 筆可搜尋／篩選清單 |
| [`ggufrun_recipe.md`](ggufrun_recipe.md) | GGUFRun 操作速查：四種情境的設定值、`-ncmoe` 掃描命令、驗檔命令、死循環急救表 |
| [`models.csv`](models.csv) | 63 筆可排序清單（執行結果／品質／速度／評語／連結核對狀態） |
| [`chart_data.csv`](chart_data.csv) | 圖表的可稽核數據（參數量／量化／每 token 估算讀取量／估算上限 vs 實測） |
| [`report_assets/`](report_assets) | 六張輔助圖（PNG） |
| [`make_charts.py`](make_charts.py) ／ [`make_dashboard.py`](make_dashboard.py) | 重繪腳本（改數據或配色重跑即可，輸出可重現） |
| [`source_llm_evaluation_4096_q4kv_report.md`](source_llm_evaluation_4096_q4kv_report.md) | 原始實測紀錄存檔（**數據一字未改**，供對照） |

### 硬體與測試條件

| 項目 | 內容 |
| :--- | :--- |
| GPU | NVIDIA RTX 3060 Ti，**8 GB GDDR6，448 GB/s** |
| RAM | **32 GB DDR4-2666**（雙通道，實務帶寬約 35～40 GB/s） |
| 可用記憶體池 | **約 30 GB ＝ 顯存 8 GB（448 GB/s）＋ RAM 約 22 GB（約 40 GB/s）**——兩層速度差 **11 倍**，這是全報告的核心 |
| 推論條件 | Context **4096**、**Q4KV**、單次任務「1,000 字推理小說」 |
| 評分 | GPT-SOL 標準（6.0 / 10 及格） |
| 樣本 | **63 筆**（主表 48 ＋ MoE/NCMoE 表 15） |

### 核心數字

| 觀察 | 數字 |
| :--- | :--- |
| 跑得完 | **41 / 63（65%）** |
| 死循環或崩潰 | **17 / 63（27%）** |
| 根本無法執行 | **5 / 63（8%）** |
| 品質被評為明確正面（A） | **2 / 63**（`gemma-4-26B-A4B-it-UD-Q2_K_XL` 本體與其 ＋Q8MTP 配置） |
| 本機最佳 GPT-SOL 分數 | **5.8**（未達 6.0 及格線） |

### 結論摘要

1. **真正的牆是記憶體帶寬，不是算力。** 顯存 448 GB/s、RAM 約 40 GB/s、SSD 約 3 GB/s；
   權重只要被擠出顯存，每讀 1 GB 的成本就多 11 倍（掉到 SSD 是 150 倍）。
2. **可用池約 30 GB（顯存 8 ＋ RAM 22），但不是均勻的池。** 「塞得下」與「跑得動」是兩個獨立關卡。
3. **大模型唯一的破局手段是 MoE ＋ `-ncmoe` 分層。** A3B/A4B 型實測可到
   **35B Q3（25 TKS）**、**48B IQ3_M（17 TKS）**、**20B Q8（22 TKS）**；dense 模型沒有這個槓桿
   （30B Q4 只剩 **1.5 TKS**）。
4. **三個極限**：① 池的硬上限 30 GB（45B Q4／70B Q3 幾乎沒有 KV 空間）；② dense 無專家可分；
   ③ 格式／架構支援性（`TQ1_0/TQ2_0` 需專用 fork、部分客製 MoE 未合併上游、個別模型與現行 CUDA build 不相容）。
5. **最大的失敗模式是死循環（27%）。** 幾乎集中在「極低量化（Q1/Q2/IQ1/IQ2）＋ 長輸出 ＋ 非中文主語料或被剪枝的模型」，
   多半是**參數與流程問題，不是模型沒救**。
6. **工具**：[GGUFRun](https://github.com/BBQ2077/GGUFRun) —— 一個視窗就能掃 `-ncmoe`、切 KV 量化、
   開投機解碼與 DRY，並把每顆模型的最佳參數存成模板。

### 六張圖

| 圖 | 說明 |
| :--- | :--- |
| ![圖 1](report_assets/01_memory_bandwidth.png) | **圖 1　記憶體階層與帶寬落差**：顯存 448 GB/s、RAM 約 40 GB/s、SSD 約 3 GB/s |
| ![圖 2](report_assets/02_bandwidth_ceiling_vs_measured.png) | **圖 2　帶寬上限 vs 實測**：63 筆散點疊上三條上限線；左上＝小而快（權重全在顯存），右下＝被擠到 RAM |
| ![圖 3](report_assets/03_size_vs_pool.png) | **圖 3　模型載入容量 vs 30 GB 池**：綠＝純顯存、藍＝MoE 分層可行、橘＝dense 大模型（容量行、速度死） |
| ![圖 4](report_assets/04_results_distribution.png) | **圖 4　63 筆結果分布**：量化位元越低，死循環比例越高 |
| ![圖 5](report_assets/05_ncmoe_tiering.png) | **圖 5　MoE ＋ NCMoE 分層配置示意** |
| ![圖 6](report_assets/06_capacity_decision.png) | **圖 6　容量決策線**：≤8 / 8–30 / >30 GB 三段 |

### 怎麼用這份專案

1. **先讀主報告第 0、1 節**：搞懂「容量 vs 帶寬」兩個瓶頸，選型就不會亂試。
2. **要挑模型 → 第 8 節決策樹**（依任務：1,000 字／3,000 字／程式／小工具）。
3. **模型一直死循環 → 第 6 節六種對策**（依成本排序，通常第 1 招就解決）。
4. **要動手跑 → `ggufrun_recipe.md` ＋ [GGUFRun](https://github.com/BBQ2077/GGUFRun)**。
5. **要重繪圖表 → `python make_charts.py`、`python make_dashboard.py`**（需 matplotlib ＋ 中文字型）。

### 資料來源與可信度

- 原始數據為作者本機實測紀錄（2026-09-18），本專案**一字未改**存檔於 `source_` 檔，
  63 筆評語與 TKS 數值全部保留，可用 `diff` 直接比對。
- 新增的分析（帶寬上限推算、分組統計、對策）皆為**由該數據推導**；估算值一律標示「估算」，
  不一致處以「可能的解釋」呈現，不當成結論。
- 參數量與量化位元由檔名推得，推不出者標為「未評」或留空，不用猜測補值。
- 連結核對：19 筆有疑點（見主報告附錄 B），**未擅自改寫任何 URL**。
- 圖表由 `make_charts.py` 從同一份數據產生，**重跑結果 byte-identical**（可重現）。

### 授權與聲明

- 本報告為個人硬體實測紀錄，**非官方 benchmark**；不同驅動、CUDA build、`llama.cpp` 版本與背景負載都會影響數字。
- 文件內容（報告、圖表）採 **CC BY 4.0**；`make_charts.py`、`make_dashboard.py` 採 **MIT**。
- [GGUFRun](https://github.com/BBQ2077/GGUFRun) 為獨立專案（MIT），本專案僅以其為操作範例。
- 文中模型名稱與連結指向各自的 Hugging Face 頁面，版權與授權依原模型發佈者為準。

---

## English

### What this is

A hands-on local-LLM benchmark report for a **budget 8 GB-VRAM rig**, measured with one realistic task:
*write a 1,000-character mystery short story* at **context 4096 with Q4 KV cache**.
**63 model configurations** were tested and scored (GPT-SOL rubric, 6.0/10 = pass).
The headline finding: on this class of hardware, **memory bandwidth — not compute — is the wall**.

### Hardware under test

| Item | Spec |
| :--- | :--- |
| GPU | NVIDIA RTX 3060 Ti — **8 GB GDDR6, 448 GB/s** |
| RAM | **32 GB DDR4-2666** (dual channel, ~35–40 GB/s real-world) |
| Usable memory pool | **~30 GB total = 8 GB VRAM + ~22 GB RAM**, but the two tiers differ by **11×** in bandwidth |
| Inference setup | context **4096**, **Q4 KV cache**, single-shot 1,000-character story task |
| Scoring | GPT-SOL rubric (6.0/10 = pass) |
| Sample size | **63 runs** (48 main + 15 MoE / NCMoE configurations) |

### Key numbers

| Observation | Result |
| :--- | :--- |
| Completed successfully | **41 / 63 (65%)** |
| Infinite loop or crash | **17 / 63 (27%)** |
| Could not run at all | **5 / 63 (8%)** |
| Rated clearly good (grade A) | **2 / 63** |
| Best local GPT-SOL score | **5.8** (below the 6.0 pass mark) |

### Findings

1. **Bandwidth is the wall, not FLOPS.** VRAM 448 GB/s vs system RAM ~40 GB/s vs SSD ~3 GB/s —
   once weights spill out of VRAM, every generated token costs ~11× more to read (150× if it reaches the SSD).
2. **The usable pool (~30 GB) is not uniform.** "It fits" and "it runs" are two separate gates.
3. **MoE + `-ncmoe` offload is the only real escape hatch.** Measured usable: **35B Q3 (25 tok/s)**,
   **48B IQ3_M (17 tok/s)**, **20B Q8 (22 tok/s)**. Dense models get no such lever —
   **30B Q4 = 1.5 tok/s** even though it fits in memory.
4. **Three hard limits**: (i) the 30 GB pool ceiling — 45B Q4 / 70B Q3 leave no room for a KV cache;
   (ii) dense models have no experts to split; (iii) format/architecture support — some quant types need
   custom forks, some custom MoE architectures are not merged upstream and simply will not run on a
   stock CUDA build.
5. **The dominant failure mode is the infinite loop (27%)**, concentrated in very low-bit quants
   (Q1/Q2/IQ1/IQ2) plus long outputs — mostly a *parameters and pipeline* problem, not a dead model.
6. **Tooling**: [GGUFRun](https://github.com/BBQ2077/GGUFRun) — a Windows GUI front-end for
   `llama-server` exposing MoE offload sliders, KV quantisation, speculative decoding, DRY sampling
   and per-model presets.

### Repository layout

| File | Contents |
| :--- | :--- |
| `3060ti_8g_model_report.md` | Full report (Traditional Chinese): hardware baseline, bandwidth cross-check, NCMoE tiers and limits, all 63 results, loop-fix playbook, decision tree, GGUFRun chapter |
| `report_dashboard.html` | Single-page dashboard with embedded charts (open directly in a browser) |
| `ggufrun_recipe.md` | GGUFRun quick-reference: four scenarios, `-ncmoe` sweep, model validation, loop first-aid |
| `models.csv` | All 63 runs, sortable (result / quality / tok/s / verdict / link status) |
| `chart_data.csv` | Auditable numbers behind the figures |
| `report_assets/` | The six figures (PNG) |
| `make_charts.py`, `make_dashboard.py` | Regenerate figures and dashboard from the same data (reproducible) |
| `source_llm_evaluation_4096_q4kv_report.md` | Original raw measurement log, archived **verbatim** |

### Caveats

- This is a **personal measurement log, not an official benchmark**: driver, CUDA build,
  `llama.cpp` revision and background load all shift these numbers.
- Documents and figures: **CC BY 4.0**. Scripts: **MIT**.
- Model names link to their respective Hugging Face pages; licensing follows each upstream publisher.

_Measured 2026-09-18 ｜ RTX 3060 Ti 8 GB + 32 GB DDR4-2666 ｜ Tooling: [GGUFRun](https://github.com/BBQ2077/GGUFRun)_
