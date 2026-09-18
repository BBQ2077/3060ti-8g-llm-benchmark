# 本機 LLM 評測報告：4096 Q4KV 推理小說生成測試

> **測試環境**：3060 Ti (8GB VRAM) + 32GB System RAM  
> **測試條件**：Context 4096, Q4KV, 目標任務「撰寫篇幅 1,000 字推理小說」  
> **評分基準**：GPT-SOL 評分標準（6.0 / 10 分為及格線）

---

## 結論先行與核心摘要

1. **陣營特性定位**：
   - **Gemma 系列**：文字奔放、詞藻生動，具備優秀的創意氛圍；但上下邏輯銜接較弱、注意力易發散（類似 Gemini Flash 特性），適合前期創意提案與發想，長篇單獨創作極易邏輯崩潰。
   - **Qwen 系列**：邏輯約束能力較強，結構完整度高於同級，適合需要一定邏輯的任務（如程式片段、局部程式碼或具體情節收斂）。
2. **硬體極限與選型建議（8GB VRAM + 32GB RAM）**：
   - **純顯存極限**：約 12B Q4（約佔 6.7GB VRAM，如 Gemma-4-12B）。
   - **NCMoE / 大參數 MoE 破局**：結合 8GB VRAM + 32GB RAM（扣除系統 10GB 後約可利用 30GB），理論可跑 45B Q4 或 70B Q3，極低量化（Q1/Q2）可上探 90B～100B。若搭配 N-gram 查找與 SSD/RAM 分層，能進一步解放容量瓶頸。
   - **8GB 顯存免折騰首選**：[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)（約 5.9GB），但需專用分支與指定參數以避免死循環；長文注意力在 2,000～3,000 字後易崩。
   - **剪枝警示**：海外社群模型常在剪枝（Pruning）過程中破壞中文語料分佈（如 REAP-320 系列），常出現英文寫作正常但中文無限死循環的現象。

---

## 模型實測評測總表

| 模型名稱 / 檔案名稱 | 核心測試評價與特徵反饋 | 推理速度 (TKS) | 參考連結 (Hugging Face / 專案) |
| :--- | :--- | :---: | :--- |
| **Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P** | 詞藻華麗但缺乏實質內容；寫滿字數不崩潰、無無限死循環。 | 77 TKS | [HauhauCS/Gemma-4-E4B-Uncensored](https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive) |
| **Bonsai-27b-1bit-CRACK-Q1_0** | 極差，中英混雜嚴重，頻繁陷入無限循環，幾乎無法使用。 | - | [dealignai/Bonsai-2-27B-1bit-CRACK](https://huggingface.co/dealignai/Bonsai-2-27B-1bit-CRACK-GGUF) |
| **Bonsai-27B-Q1_0 (官方版)** | 表現同破解版，中英混雜且無限循環。 | - | [prism-ml/Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) |
| **Bonsai-27B-dspark-dflash-Q4_0** | 0.6G DF 配置，運行速度顯著下降。 | 慢速 | [prism-ml/Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) |
| **Huihui-MiniCPM5-2B-abliterated-GGUF** | 體積極小且速度飛快，但創意性極低，定位偏向數理小工具模型。 | 120 TKS | [huihui-ai/MiniCPM5-2B-abliterated](https://huggingface.co/huihui-ai/Huihui-MiniCPM5-2B-abliterated) |
| **Huihui-Spark-X2.5-4B-abliterated-Q4_K_M** | 帶有 Gemma 系列風格，在此參數級別表現尚可，超越同級小鋼炮。 | 100 TKS | [huihui-ai/Spark-X2.5-4B-abliterated](https://huggingface.co/huihui-ai/Huihui-Spark-X2.5-4B-abliterated) |
| **K2-Horizon-7B-Q4_K_M.gguf** | 與當前 llama-b11002-bin-win-cuda 環境不相容，無法執行。 | - | [IFM/K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B) |
| **Ornith-1.5-9B-GGUF** | 介於普通作文與拙劣之間，表現微妙。 | 60 TKS | [mradermacher/Ornith-1.5-9B-uncensored](https://huggingface.co/mradermacher/Ornith-1.5-9B-uncensored-GGUF) |
| **Qwen3.8-4B-Distill-GGUF** | 優於 Gemma-4-E4B，呈現初階作者筆觸。 | 100 TKS | [empero-ai/Qwen3.8-4B-Distill](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF) |
| **maple-preview-TQ1_0-head-Q4_K** | 文筆優秀，比 Qwen3.8-4B 好，但故事結構欠佳（長文無大綱），3000 字後邏輯渙散但無單句死循環。 | 38 TKS | [deepgrove/maple-preview](https://huggingface.co/deepgrove/maple-preview) |
| **gemma4-v2-Q3_K_M-yuxinlu1** | 量化過度導致崩潰，陷入無限迴圈。 | - | [yuxinlu1/gemma-4-12B-agentic-v2](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF) |
| **Huihui-Qwen3-8B-abliterated-v2-Q4_K_M-GGUF** | 虎頭蛇尾，篇幅拉長即出現死循環。 | 70 TKS | [huihui-ai/Qwen3-8B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) |
| **Qwen3.8-27B RVN Heretic Abliterated-IQ1_S** | 觸發無限迴圈後自動中斷。 | 30 TKS | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Qwen3.8-9B-Q4_K_M.gguf** | 行文風格酷炫但完全缺乏邏輯內核；但能完整寫完不卡死。 | 67 TKS | [empero-ai/Qwen3.8-9B-Distill](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF) |
| **Ling-3.0-tiny-Q4_K_M.gguf** | 陷入無限迴圈。 | 158 TKS | [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) |
| **LFM2.5-2.6B-Q4_K_M.gguf** | 劇情尚可，但強制挪用給定開頭且人物隨機更換，穩定性差。 | 164 TKS | [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| **Phi-4-mini-instruct-Q4_K_M.gguf** | 容易無限循環；偶爾生成但離題且品質低劣。 | 116 TKS | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| **Qwen3.6-14B-A3B-FableVibes-Q2_K.gguf** | 無限死循環，且非預期切換為英文。 | 85 TKS | [tvall43/Qwen3.6-14B-A3B-FableVibes](https://huggingface.co/tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF) |
| **Qwen3-8B-Q4_K_M.gguf (官方版)** | 故事僅寫 600 字開頭；要求 3000 字時直接回覆超出能力上限。 | 70 TKS | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| **OxCoder-9B.Q4_K_M.gguf** | 1000 字能完結但有邏輯硬傷；拉長至 3000 字會無限循環。 | 67 TKS | [OrionLLM/OxCoder-9B](https://huggingface.co/OrionLLM/OxCoder-9B) |
| **Nanbeige4.2-3B-Q4_K_M.gguf** | 開頭表現可接受，但篇幅拉長立即死循環。 | 60 TKS | [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) |
| **Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced** | 氛圍塑造佳，但推理邏輯薄弱。 | 12 TKS | [HauhauCS/Gemma4-12B-QAT](https://huggingface.co/google/gemma-4-12B-it) |
| **gemma-4-12b-it-UD-Q2_K_XL.gguf** | 無限迴圈。 | 43 TKS | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |
| **glm-4-9b-chat.Q4_K_S.gguf** | 完全離題，輸出政治新聞相關內容。 | 50 TKS | [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) |
| **Mistral-7B-Instruct-v0.3.Q4_K_M.gguf** | 缺乏邏輯推演，非推理小說結構。 | 16 TKS | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| **Qwen3.5-9B-The-Defiant-Fable-Uncnr-IQ3_M** | 偏離推理題材，寫成超自然奇幻小說。 | 24 TKS | [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen) |
| **maple-tq2_0.gguf** | 依賴特殊客製化 llama.cpp 分支始能加載。 | - | [deepgrove/maple-preview](https://huggingface.co/deepgrove/maple-preview) |
| **Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M** | 無限迴圈卡死。 | 10 TKS | [mradermacher/Qwythos-9B](https://huggingface.co/mradermacher) |
| **Qwen3.8-27B-UD-IQ1_S.gguf** | 輸出僅約 50 字即異常終止。 | 9 TKS | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf** | 前半段優秀，但後續結構嚴重發散且無結局。 | 35 TKS | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| **gemma-4-12B-it-qat-UD-Q4_K_XL.gguf** | 創意度高，唯存在常規邏輯瑕疵。 | 13 TKS | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |
| **gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf** | 表面詞藻與結構優秀，唯核心推理環節矛盾。 | 65 TKS | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |
| **Hermes-3-Llama-3.1-8B.Q4_K_M.gguf** | 偏向冒險小說，表現平庸但輸出穩定無短板，適合評估作代理模型。 | 50 TKS | [NousResearch/Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B) |
| **DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q4_K_M.gguf** | 開頭尚可，中後段陷入無限死循環。 | 30 TKS | [deepseek-ai/DeepSeek-V2.5](https://huggingface.co/deepseek-ai) |
| **DeepSeek-Coder-V2-Lite-Instruct-IQ2_M.gguf** | 結構混亂，邏輯完全偏移。 | 70 TKS | [deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) |
| **DeepSeek-R1-Distill-Qwen-14B-IQ2_M.gguf** | 前半尚可，篇幅拉長後推理智力驟降並進入死循環。 | 25 TKS | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) |
| **DeepSeek-R1-Distill-Qwen-32B-IQ2_M.gguf** | 受限於硬體記憶體交換，速度過慢不具實用性。 | 2 TKS | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| **internlm3-8b-instruct-q4_k_m.gguf** | 偏離推理主題，生成為常規冒險小說。 | 67 TKS | [internlm/internlm2_5-7b-chat](https://huggingface.co/internlm) |
| **Yi-1.5-9B-Chat-Q4_K_M.gguf** | 直白乏味，無伏筆且邏輯前後矛盾。 | 67 TKS | [01-ai/Yi-1.5-9B-Chat](https://huggingface.co/01-ai/Yi-1.5-9B-Chat) |
| **aya-expanse-8b-Q4_K_M.gguf** | 筆觸偏國中生作文，排版格式跑版。 | 70 TKS | [CohereLabs/aya-expanse-8b](https://huggingface.co/CohereLabs/aya-expanse-8b) |
| **Qwen2.5-7B-Instruct-Q4_K_M.gguf** | 生成內容完全混亂失序。 | - | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| **Qwen3-14B-Claude-4.5-Opus-Distill.q3_k_s.gguf** | 能夠完整產出，但邏輯矛盾點較多。 | 25 TKS | [Qwen/Qwen2.5-14B](https://huggingface.co/Qwen) |
| **TieFighter-Holodeck-20B-IQ2_M-imat.gguf** | 中文支援不良，輸出亂碼。 | 14 TKS | [DavidAU/TieFighter-Holodeck](https://huggingface.co/DavidAU/TieFighter-Holodeck-Holomax-Mythomax-F1-V1-COMPOS-20B-gguf) |
| **Moonlight-16B-A3B-Instruct-IQ2_XS.gguf** | 生成亂碼。 | - | [moonshotai/Moonlight-16B-A3B-Instruct](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct) |
| **amd.Instella-MoE-16B-A3B-Think-Q2_K.gguf** | 暫無法測試（llama.cpp Instella-MoE PR #26467 尚未合併）。 | - | [amd/Instella-MoE-16B-A3B-Think](https://huggingface.co/amd/Instella-MoE-16B-A3B-Think) |
| **bohf-12b-moe-3a-q4_k_m.gguf** | 無限死循環，未能輸出小說內容。 | - | [theprint/Bohf-12B-MoE-3A](https://huggingface.co/theprint/Bohf-12B-MoE-3A) |
| **gemma-4-E4B-it-OBLITERATED.i1-Q4_K_S** | 用詞靈活奔放，但後半段邏輯崩潰。 | 84 TKS | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |
| **gemma-4-26B-A4B-it-UD-Q2_K_XL** | 整體完成度高、大幅躍進，被 GPT-SOL 評為最像樣版本；但仍存在細部邏輯瑕疵。 | 10 TKS (思考: 5 TKS) | [google/gemma-4-26B](https://huggingface.co/google) |

---

## 大參數 MoE 與 NCMoE 突破測試

| 模型架構 / 運行規格 | 測試反饋與重點結論 | 速度 | 參考連結 |
| :--- | :--- | :---: | :--- |
| **gemma-4-26B-A4B-it-UD-Q2_K_XL + Q8MTP (n_max=1)** | 支援 32K 上下文、Q4KV 配置，速度表現良好。 | 35 TKS | [google/gemma-4-26B](https://huggingface.co/google) |
| **Qwen3.6-35B-A3B-UD-Q3_K_XL** | 無死循環問題，文筆具備初階作者水準，但有細微邏輯瑕疵。 | 25 TKS | [Qwen/Qwen3.6-35B](https://huggingface.co/Qwen) |
| **gpt-oss-20b-Q8_0** | 無死循環，行文偏平庸（國中生水準），GPT-SOL 評分低於 Qwen。 | 22 TKS | [openai/gpt-oss](https://huggingface.co/openai) |
| **ERNIE-4.5-21B-A3B-Thinking-UD-Q4_K_XL** | 解析格式異常，輸出內容直接被導向 Log 檔。 | - | [baidu/ERNIE-4.5](https://huggingface.co/baidu) |
| **Nemotron-3-Nano 30B-A3B Q4_0** | 核心推理邏輯存在矛盾。 | 12 TKS | [nvidia/Nemotron-3-Nano](https://huggingface.co/nvidia) |
| **GLM-4.7-Flash-UD-Q3_K_XL** | 缺乏真正的推理內核，呈現偽推理結構。 | 30 TKS | [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM) |
| **Ornith-1.5-35B-A3B-IQ4_XS** | 文風中庸，存在邏輯瑕疵，特性偏 Coding/Agent。 | 37 TKS | [ornith-ai/Ornith-1.5-35B](https://huggingface.co/ornith-ai) |
| **Ternary-Bonsai-2-27B-PTQ1_0** | 8G 顯存極限；需專用 fork 與特定參數才能抑止循環。長文 10,000 字測試在 2,000～3,000 字處崩潰。 | 28 TKS | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) |
| **Qwen3.8-Flash-Next-UD-Q2_K_XL-reap320** | 中文寫作陷入無限迴圈；英文寫作正常流暢。剪枝嚴重破壞中文語料能力，且欠缺真正推理內核。 | 10 TKS | [Litwein/Qwen3.8-Flash-Next-REAP320](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MTPLX) |
| **K2-Horizon-MoVA-36B-A4B-Q3_K_M** | 陷入死循環，過程中夾雜跳出英文。 | 13 TKS | [IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) |
| **Qwen3.8-Flash-Next-125B-UltraLite-37GiB** | 需額外編譯，簡單中英問題均無法正常回應，表現極差。 | 6 TKS | [AnonimousA/Qwen3.8-Flash-Next-REAP-320](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF) |
| **nex-agi_Nex-N2.5-mini-IQ2_M** | 表現類似 Qwen3.6-35B，存在核心邏輯矛盾。 | 39 TKS | [nex-agi/Nex-N2.5-mini](https://huggingface.co/nex-agi/Nex-N2.5-mini) |
| **Huihui4-48B-A4B-abliterated.i1-IQ3_M** | 文筆生動，但情節邏輯荒謬。 | 17 TKS | [huihui-ai/Huihui4-48B-A4B](https://huggingface.co/huihui-ai) |
| **JoyAI-LLM-Flash-IQ3_XS** | 存在些許相容問題，核心邏輯大矛盾，但具備類似 Gemma 的鮮明文字風格。 | 21 TKS | [jdopensource/JoyAI-LLM-Flash](https://huggingface.co/jdopensource/JoyAI-LLM-Flash) |
| **Muse-Glimmer-30B-UD-Q4_K_XL** | 運算耗時極長，速度過慢不具實用性。 | 1.5 TKS | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) |

---

## GPT-SOL 基準評分榜 (6.0 為及格線)

```
[雲端旗艦級別]
├── Gemini 3.8 Flash Med       : 7.8 分 (及格，推理能力佳)
├── GPT-SOL High               : 7.4 分 (及格)
└── Opus 4.6 High              : 7.2 分 (及格)

[本機開源級別]
├── Qwen3.8 FN Q1              : 5.8 分 (接近及格，邏輯較整齊)
├── DeepSeek-4.1 Flash         : 5.0 分 (推理小說標準不及格，寫成懸疑小說)
├── Nex-N2.5-mini (IQ2_M)      : 4.8 分 (43 TKS)
├── Ternary-Bonsai-2-27B       : 2.2 ~ 5.1 分 (浮動劇烈；無自我收斂機制，思考越多錯誤越多)
├── JoyAI-LLM-Flash (IQ3_XS)   : 3.4 分 (21 TKS)
└── Huihui4-48B-A4B (IQ3_M)    : 2.8 分 (17 TKS)
```

---

## 參數梯隊選型指南（8GB VRAM + 32GB RAM）

| 參數級別 | 能力定位與特徵 | 建議與代表模型 | 適用場景 |
| :--- | :--- | :--- | :--- |
| **$\le$ 5B** | 小學生作文水準或功能型小工具 | • [MiniCPM5-2B](https://huggingface.co/huihui-ai/Huihui-MiniCPM5-2B-abliterated)<br>• [Spark-X2.5-4B](https://huggingface.co/huihui-ai/Huihui-Spark-X2.5-4B-abliterated) | 數學運算、小工具函式生成 |
| **6B ~ 7B** | 尷尬過渡區，上下不討好 | 多數舊架構 7B | 不建議特別投入 |
| **8B ~ 12B** | 國中生作文，能堆疊華麗詞藻但缺乏實質內核 | • [Gemma-4-E4B-it](https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive)<br>• [Gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it)<br>• [Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B) | 8GB 顯卡純 VRAM 極限區；代理 Agent、短文發想 |
| **12B ~ 25B** | 具備一定全面性，但缺乏大模型質變感 | • [gpt-oss-20b](https://huggingface.co/openai)（全面但平庸） | 綜合一般對話 |
| **26B ~ 30B** | 展現高中生思考水準，甜點區間 | • [Gemma-4-26B-A4B-it](https://huggingface.co/google)<br>• [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)<br>• [Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (5.9GB 適合純 8G VRAM) | 創意長文雛形、邏輯程式碼生成 |
| **30B ~ 50B** | 大學生思考水準，主力工作梯隊 | • [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B) (主力推薦)<br>• [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai)<br>• [Nex-N2.5-mini](https://huggingface.co/nex-agi/Nex-N2.5-mini) | 複雜推理任務、程式輔助、結構性創作 |
| **50B ~ 100B** | 甜蜜點真空區 | 缺乏 27B/35B 般明確甜點模型 | 8G VRAM + 32G RAM 較少受益 |
| **100B+** | 大模型旗艦區 | • [Qwen3.8-Flash-Next](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF) (125B LM / 6B active)<br>• [AnonimousA REAP-320](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MTPLX) (剪枝版) | 英文/程式/Agent 架構（需注意剪枝版中文易崩潰） |
