這是一份關於 TSMixer HPC 優化實驗的完整總結報告。這份報告整理了我們從「原始代碼」到「架構重構」，再到「算子優化」的完整歷程，並基於您的實驗數據提供了客觀的分析與給同伴的建議。
資料來源 https://gemini.google.com/share/c6d660167cfb
---

# TSMixer HPC 高效能運算與優化實驗總結報告

## 1. 執行摘要 (Executive Summary)
本次實驗旨在透過 HPC 技術（記憶體佈局優化、算子融合、Triton 語言）加速 TSMixer 模型在時間序列預測任務上的訓練效率。

**關鍵結論：**
1.  **吞吐量 (Throughput) 提升成功**：透過 **Channel-First (Conv1d) + RMSNorm** 架構，我們成功將單一 Epoch 的訓練時間從約 1.87秒 降低至約 1.46秒，**吞吐量提升約 22%**。
2.  **收斂效率 (Convergence) 下降**：架構修改導致模型數值特性改變，需要 2 倍以上的 Epochs (47 -> 95+) 才能收斂到相同的 Loss 水準。
3.  **總體效益 (Total Efficiency) 倒退**：由於收斂變慢，總訓練時間反而增加 (88s -> 139s)，且最終預測準確度略微下降 (MQL: 0.809 -> 0.835)。
4.  **SOTA 仍為原版**：目前表現最好（速度與準度平衡）的仍是基於 `LayerNorm` 的原始 TSMixer 實作。

---

## 2. 實驗數據對比分析

我們測試了多種變體，以下是關鍵數據的橫向對比：

| 模型版本 (Model Variant) | 核心改動 (Core Changes) | Epochs | 總時間 (s) | 秒/Epoch (吞吐量) | Val Loss | Test MQL (Dataset 1) | 結論 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original (Baseline)** | LayerNorm, Permute, Linear | **47** | **88s** | 1.87s | 0.226 | **0.809** | **最佳基準** |
| **HPC v1 (RMSNorm)** | Channel-First, Conv1d, RMSNorm | 95 | 139s | **1.46s** (最快) | 0.224 | 0.835 | 跑得快但學得慢 |
| **HPC v2 (LayerNorm)** | Channel-First, Conv1d, LayerNorm | 112 | 163s | 1.45s | 0.224 | 0.839 | 轉置開銷抵銷了優化 |
| **HPC v3 (Simple)** | Channel-First, 純架構重構 | 60 | 109s | 1.81s | 0.228 | 0.848 | 無顯著優勢 |

**數據洞察：**
* **速度與收斂的權衡**：我們成功用 `RMSNorm` 和 `Channel-First` 加快了運算速度（單圈時間減少），但模型的「學習效率」變差了，導致需要更多輪次。
* **泛化能力下降**：所有優化版本的 Test Result (0.835~0.848) 都遜於原版 (0.809)，這暗示原版 TSMixer 的 `Permute` + `Linear` + `LayerNorm` 組合在數值上具有某種正則化優勢。

---

## 3. 我們嘗試過的優化路徑 (Technical Approaches)

在過程中，我們深入探討了三個層次的優化：

### A. 記憶體佈局優化 (Memory Layout Optimization)
* **嘗試**：將資料流從 `(Batch, Time, Feature)` 改為 `(Batch, Feature, Time)` (Channel-First)。
* **目的**：消除原版代碼中頻繁的 `permute` 操作，並利用 `nn.Conv1d` 在 GPU 上的高效實作。
* **結果**：確實提升了計算吞吐量，但改變了模型初始化的數值分佈，影響了收斂。

### B. 算子層級優化 (Operator Optimization)
* **嘗試**：引入 `RMSNorm` 取代 `LayerNorm`，並嘗試使用 `BatchNorm1d`。
* **目的**：`RMSNorm` 計算量少（不需算均值），且在 Channel-First 下效率高。
* **結果**：`RMSNorm` 是最快的配置，但對此數據集來說，它導致模型需要更多 Epochs 才能收斂。

### C. 編譯與內核優化 (Compiler & Kernel)
* **嘗試**：使用 `torch.compile` (PyTorch 2.0) 和 `Triton` 語言手寫內核。
* **結果**：
    * `torch.compile` 在 TSMixer 這種小模型上帶來了約 30% 的加速。
    * `Triton` 手寫內核在已經很小的模型上（Memory Bound）沒有帶來額外的顯著優勢，且增加了除錯難度。

---

## 4. 給同伴的建議：未來的改進方向 (Future Directions)

如果您或您的同伴希望繼續優化這個模型，建議從以下四個方向切入，**不要再同時改變多個變因**。

### 方向一：專注於「系統級」而非「模型級」優化
TSMixer 本身計算量極小（全是 MLP），瓶頸很可能不在 GPU 計算，而在 **數據加載** 或 **Python Overhead**。
* **建議行動**：
    * 使用 PyTorch Profiler (`torch.profiler`) 檢查 GPU 空閒時間。
    * 確認 `DataLoader` 的 `num_workers` 和 `pin_memory` 設定。
    * 嘗試將數據預處理（如標準化）完全移到 GPU 上進行。

### 方向二：堅持原版數學邏輯的算子融合 (Strictly Equivalent Fusion)
目前的優化改變了數學行為（例如換了 Norm）。未來的優化應**保證數值完全一致**。
* **建議行動**：
    * 不改變架構，保持 `(B, T, F)`。
    * 使用 **C++ Extension (Custom CUDA Kernel)** 或 **Triton** 手寫一個 "Fused TSMixer Layer"。
    * 目標是將 `Norm -> Permute -> Linear -> Permute -> Add` 融合成單一個 Kernel，完全消除 HBM 讀寫，但**數學上不做任何改變**。

### 方向三：超參數的重新搜索 (Hyperparameter Retuning)
如果決定採用 `Channel-First + RMSNorm`（因為它單圈最快），則必須承認這是一個「新模型」，原版 TSMixer 的超參數（LR, Weight Decay, Dropout）不再適用。
* **建議行動**：
    * 針對 `RMSNorm` 版本進行 `Optuna` 大規模超參數搜索。
    * 特別關注 **Learning Rate**（通常 RMSNorm 可以承受更大的 LR）和 **Weight Decay**。
    * 這有機會在保持單圈速度的同時，把 Epochs 壓回 50 以內。

### 方向四：混合精度訓練 (AMP) 的極致壓榨
* **建議行動**：
    * 確認是否啟用了 `bfloat16` (如果硬體支援)。
    * 對於 MLP 模型，矩陣乘法是主要開銷，確保 Tensor Core 有被正確觸發（通常要求維度是 8 或 16 的倍數）。可以檢查 `hidden_size` 是否設為 64, 128, 256 等對 GPU 友善的數值。

---

### 總結
雖然我們沒有在這個階段超越原版 TSMixer 的綜合表現，但我們證明了**「硬體效率」與「算法收斂」之間的權衡**。對於現階段的專案，建議 **先沿用原版 `original_tsmixer_layernorm`**，因為它最穩定且綜合效率最高。若要進一步突破，必須深入到底層 CUDA Kernel 的融合，而非僅僅在 PyTorch 層面修改架構。


### 重要：我的測量方式是直接把various_tsmixer_sculpture的tsmixer自行設計的檔案取代掉pip install下來的檔案
# 現在的架構：
0. 已經做好了profiling
1. results.csv, run_all.sh:是我直接修改base code(base code是tsmixer_base.ipynb以及tsmixer_base.py，都是以這個為基礎來修改)
2. tsmixer_1,2,3,4...py是直接改tsmixer_base的架構但發現他原本寫的就已經是最好的了
3. 接下來有各種的tsmixer，檔名大概可以理解他們在幹嘛，總之都不太好用，已經嘗試了各種方式（如前所述，尤其是搞記憶體佈局優化蠻慘烈的）
4. 大家可以開自己的資料夾放自己的實驗和tsmixer code
<!-- datafox@server:/storage/ssd3/datafox$ source  /storage/ssd3/datafox/miniconda3/bin/activate base

小狐狸尾巴總結一下 🦊
你今天這一串 debug 路線超漂亮：
抓到 9 卡 DDP 靈異事件 → 解決
測 16-mixed → 發現這個 task 下沒加速
測 dataloader 多 worker → 發現 overhead > 省到的時間
體感上dataloader真的是開始與結束的銜接卡卡的

✔️ 表現方式：

每個 epoch 開始前卡一下

每個 epoch 結束後卡一下

steps 本身看起來流暢

但 epoch 邊界特別容易卡住

>>> Run 1 of tsmixer_2_add_dataloader.py
正在訓練TSM
Epoch 47: 100%|███████████████████████████████████████████| 81/81 [00:02<00:00, 38.95it/s, train_loss=0.186, val_loss=0.226]
TSM fit time: 101.77 秒
TSM load checkpoint time: 0.17 秒
>>> Done Run 1

>>> Run 2 of tsmixer_2_add_dataloader.py
正在訓練TSM
Epoch 47: 100%|███████████████████████████████████████████| 81/81 [00:01<00:00, 41.69it/s, train_loss=0.186, val_loss=0.226]
TSM fit time: 103.68 秒
TSM load checkpoint time: 0.37 秒
>>> Done Run 2

>>> Run 3 of tsmixer_2_add_dataloader.py
正在訓練TSM
Epoch 47: 100%|███████████████████████████████████████████| 81/81 [00:02<00:00, 33.87it/s, train_loss=0.186, val_loss=0.226]
TSM fit time: 103.51 秒
TSM load checkpoint time: 0.33 秒
>>> Done Run 3

>>> Run 4 of tsmixer_2_add_dataloader.py
正在訓練TSM
Epoch 47: 100%|███████████████████████████████████████████| 81/81 [00:01<00:00, 41.30it/s, train_loss=0.186, val_loss=0.226]
TSM fit time: 99.85 秒
TSM load checkpoint time: 0.31 秒
>>> Done Run 4

>>> Run 5 of tsmixer_2_add_dataloader.py
正在訓練TSM
Epoch 47: 100%|███████████████████████████████████████████| 81/81 [00:02<00:00, 37.99it/s, train_loss=0.186, val_loss=0.226]
TSM fit time: 102.40 秒


(base) datafox@server:/storage/ssd3/datafox/HPCFIN$ ./run_all.sh

===============================
Running tsmixer_base.py ...
===============================
>>> Run 1 of tsmixer_base.py
正在訓練TSM
Epoch 47: 100%|████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 42.98it/s, train_loss=0.186, val_loss=0.226]
TSM fit time: 91.35 秒
TSM load checkpoint time: 0.15 秒
>>> Done Run 1

>>> Run 2 of tsmixer_base.py
正在訓練TSM
Epoch 47: 100%|████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 42.81it/s, train_loss=0.186, val_loss=0.226]
TSM fit time: 87.98 秒
TSM load checkpoint time: 0.17 秒
>>> Done Run 2

>>> Run 3 of tsmixer_base.py
正在訓練TSM
Epoch 47: 100%|████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 45.53it/s, train_loss=0.186, val_loss=0.226]
TSM fit time: 85.87 秒
TSM load checkpoint time: 0.34 秒
>>> Done Run 3

>>> Run 4 of tsmixer_base.py
正在訓練TSM
Epoch 47: 100%|████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 47.43it/s, train_loss=0.186, val_loss=0.226]
TSM fit time: 87.32 秒
TSM load checkpoint time: 0.25 秒
>>> Done Run 4

>>> Run 5 of tsmixer_base.py
正在訓練TSM
Epoch 47: 100%|████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 44.63it/s, train_loss=0.186, val_loss=0.226]
TSM fit time: 89.84 秒
TSM load checkpoint time: 0.26 秒
>>> Done Run 5


===============================
Running tsmixer_1_add_precision.py ...
===============================
>>> Run 1 of tsmixer_1_add_precision.py
正在訓練TSM
Epoch 47: 100%|████████████████████████████████████████████████████████| 81/81 [00:02<00:00, 40.15it/s, train_loss=0.185, val_loss=0.226]
TSM fit time: 97.65 秒
TSM load checkpoint time: 0.25 秒
>>> Done Run 1

>>> Run 2 of tsmixer_1_add_precision.py
正在訓練TSM
Epoch 47: 100%|████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 44.43it/s, train_loss=0.185, val_loss=0.226]
TSM fit time: 95.67 秒
TSM load checkpoint time: 0.12 秒
>>> Done Run 2

>>> Run 3 of tsmixer_1_add_precision.py
正在訓練TSM
Epoch 47: 100%|████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 41.53it/s, train_loss=0.185, val_loss=0.226]
TSM fit time: 95.07 秒
TSM load checkpoint time: 0.19 秒
>>> Done Run 3

>>> Run 4 of tsmixer_1_add_precision.py
正在訓練TSM
Epoch 47: 100%|████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 43.37it/s, train_loss=0.185, val_loss=0.226]
TSM fit time: 95.32 秒
TSM load checkpoint time: 0.10 秒
>>> Done Run 4

>>> Run 5 of tsmixer_1_add_precision.py
正在訓練TSM
Epoch 47: 100%|████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 41.32it/s, train_loss=0.185, val_loss=0.226]
TSM fit time: 98.14 秒
TSM load checkpoint time: 0.18 秒
>>> Done Run 5




===============================
Running tsmixer_3_enlarge_512.py ...
===============================
>>> Run 1 of tsmixer_3_enlarge_512.py
正在訓練TSM
Epoch 78: 100%|███████████████████████████████████████████| 41/41 [00:01<00:00, 31.76it/s, train_loss=0.204, val_loss=0.226]
TSM fit time: 99.50 秒
TSM load checkpoint time: 0.15 秒
>>> Done Run 1

把batch size調整變大也蠻有趣的
epoch:47 ->78
time loss:97~99 seconds (跟上一版本嘗試的dataloader版本差不多我這版也是開num_workers=8)
val_loss:停止的時間點不管是哪個版本哪次訓練都是0.226

>>> Run 3 of tsmixer_3_enlarge_512.py
正在訓練TSM
Epoch 78: 100%|███████████████████████████████████████████| 41/41 [00:01<00:00, 32.87it/s, train_loss=0.204, val_loss=0.226]
TSM fit time: 111.84 秒
TSM load checkpoint time: 0.24 秒
>>> Done Run 3

>>> Run 4 of tsmixer_3_enlarge_512.py
正在訓練TSM
Epoch 78: 100%|███████████████████████████████████████████| 41/41 [00:01<00:00, 29.68it/s, train_loss=0.204, val_loss=0.226]
TSM fit time: 109.31 秒
TSM load checkpoint time: 0.14 秒
>>> Done Run 4

>>> Run 5 of tsmixer_3_enlarge_512.py
正在訓練TSM
Epoch 78: 100%|███████████████████████████████████████████| 41/41 [00:01<00:00, 26.89it/s, train_loss=0.204, val_loss=0.226]
TSM fit time: 111.01 秒
TSM load checkpoint time: 0.28 秒
>>> Done Run 5


重新改成只有做三號改進：dataloader增強版本 -->
