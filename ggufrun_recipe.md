# GGUFRun 操作速查（3060 Ti 8G ＋ 32G DDR4-2666 專用）

> 對應報告：[`3060ti_8g_model_report.md`](3060ti_8g_model_report.md) ｜ 工具：[GGUFRun v2.0](https://github.com/BBQ2077/GGUFRun)（Windows／MIT）

---

## 1. 安裝（5 分鐘）

| 步驟 | 動作 | 檢查點 |
| :--- | :--- | :--- |
| 1 | 下載 `llama.cpp` 的 **CUDA Windows build**，把 `llama-server.exe` ＋ 所有 `.dll` 放進 `runtime\` | 也可以放 `runtime-cuda\`、`runtime-cpu\` 並存做對照，下拉選單會全部列出 |
| 2 | 把 `.gguf` 放到 `gguf-ui.py` 同一層（草稿模型也放一起） | 程式會自動列出，不需匯入 |
| 3 | 雙擊 `start-ui.bat` | 會自動探測可用的 Python（PATH → 常見安裝路徑） |
| 4 | 選模型 → `▶ 啟動` | 啟動前按 **dry-run** 看一眼完整指令 |
| 5 | `🌐 開網頁（輕量）` 開測試視窗 | 獨立 Chromium 設定檔，不共用平日瀏覽器資料 |

## 2. 本機四種情境的設定（照抄）

| 情境 | ctx | KV | -ngl | -ncmoe | 思考 | 備註 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **8～9B Q4 快速短文**（Gemma-4-E4B、Qwen3.8-9B） | 4096 | q4_0 / q4_0 | 99 | 空 | 關 | 60～84 TKS 區間 |
| **26B-A4B 追品質** | 4096 | q4_0 / q4_0 | 99 | 掃描決定 | 關 | 掛 Q8MTP 草稿模型、`n_max=1` → 10 拉到 35 TKS |
| **35B-A3B 主力** | 4096 | q4_0 / q4_0 | 99 | 12～24 之間 | 關 | 實測 25 TKS、無死循環 |
| **長文重複急救** | 4096 | 先 q4_0，再試 q8_0 | 99 | 不變 | 關 | 同時開 DRY（multiplier 0.8 / base 1.1 / allowed length 2）、min-p 0.08 |

## 3. 掃 `-ncmoe` 甜蜜點（本機最重要的一步）

```bat
set MODEL=Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
set RT_DIR=runtime
set PORT=18435
python tools\_bench_ncmoe.py 0 8 12 16 20 24 off
```

**判讀**：取「速度已接近平台期、但載入時間還沒暴增」的值。再加下去會把**活躍**專家也推到 RAM，速度反而掉。

## 4. 上機前驗檔

```bat
python tools\_arch.py  Model.gguf    :: 架構／層數／專家數／context 長度
python tools\_ttype.py Model.gguf    :: 「Q4_K_M」裡實際裝了哪些量化型別
```

用途：把「不相容」拆成兩種情況——**架構不支援**（換 runtime）或**檔案本身有問題**（換檔案）。

## 5. 下載大檔（並行分段、可續傳）

```bat
python pardl.py <URL> Model.gguf 8
```

## 6. 死循環急救對照表

| 症狀 | 先做 | 再做 |
| :--- | :--- | :--- |
| 中英夾雜跳針 | 確認量化級別不是 Q1/Q2/IQ1/IQ2 → 換 Q3/Q4 | 檢查是否為 `reap`／剪枝版（中文語料已破壞，直接換檔） |
| 思考型模型繞圈 | 關思考或 budget 512～1024 | 降低 reasoning effort |
| 1,000 字後開始重複 | 開 DRY ＋ min-p 0.08 | 改「大綱 → 分段」流程（不要正面硬拼長度） |
| 每句都重複同一段 | 檢查 repetition penalty（別超過 1.15） | 試 KV 改 q8_0；確認 ctx 沒有開過大 |

## 7. 安全須知（v2.0）

- 只綁 **`127.0.0.1`**，外網／區網連不到。
- 刪除模型 → **資源回收筒**（可還原）。
- 設定檔與瀏覽器設定檔全部留在程式資料夾。
- 關窗只關**自己啟動的那個 PID**（`atexit` 保底）。
- 啟動腳本的**命令注入**已在 v2.0 修補（惡意檔名／參數無法再逃逸執行）。
