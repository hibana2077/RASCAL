**一句話 mismatch（先講清楚）**
在 **image classification 的 supervised contrastive learning** 中，SupCon 會把 batch 內所有同類樣本視為**可交換、等權的 positives**（uniform all-to-all attraction），但真實影像類別常是**多模態且長尾**（姿態、背景、亞型、拍攝條件差異很大），因此很多同類配對其實不該被一樣強度地拉近。 

---

## 論文題目（提案）

**RASCAL: Rank-Agreement Supervised Contrastive Alignment Loss**
（中文：**以鄰域排序穩定度對齊類內結構的監督式對比損失**）

> 核心只有一個：**一個新 loss（RASCAL loss）**，直接取代 SupCon 的等權正樣本平均；**沒有額外 loss、沒有 λ 聚合、沒有新參數模組**。

---

## 研究核心問題

SupCon 的標準做法（Eq.2）對每個 anchor 將所有同類 positive 平均處理，這在理論上很容易把類內表徵推向更強的塌縮幾何（例如類別級 simplex / neural-collapse 方向），但實務上真實影像類別常包含穩定存在的子結構（sub-modes），過度等權拉近會壓扁這些子結構，影響細粒度辨識、長尾尾類辨識與 corruption robustness。 

---

## 研究目標

設計一個**輕量插拔式**、只使用模型內免費訊號的 SupCon 替代目標，使模型仍保有對比學習的 alignment / uniformity 優勢，但能**保留類內多模態結構**，特別提升：

1. 細粒度分類（子結構保留）
2. 長尾分類（尾類不被頭類/同類異模態牽引過度）
3. 常見 corruption 下穩健性（結構一致性更穩）
   ([Proceedings of Machine Learning Research][1])

---

## 貢獻

1. **提出 RASCAL loss（唯一核心貢獻）**：用「**類內鄰域排序穩定度**」取代 SupCon 中對 positives 的等權平均。
2. **結構化對齊目標**：把免費訊號轉成一個 **anchor-conditioned 非對稱 rank graph**（row-stochastic），並在單一 loss 中完成對齊。
3. **理論可解釋性**：給出 loss 分解（正樣本總量 + rank 分布一致性）、梯度形式與退化到 SupCon 的條件。
4. **零額外參數量**：只需一個 detached feature buffer（或上一輪 embedding cache），訓練/推論參數不變。

---

## 創新

不是做 hard positive mining，也不是再加 prototype / routing head / 多損失權重；而是直接把 **「哪些同類關係在訓練過程中是穩定的」** 當成結構訊號，轉成一個**可對齊的 rank target**，嵌進 SupCon 本體。這比「看當下 similarity 大小」更接近真實類內結構，因為它利用了 training dynamics 的穩定性資訊（但不用任何額外標註）。

---

## 理論洞見

SupCon 已知強調正對齊與全局分布均勻性（alignment / uniformity），也常伴隨更強的類別級塌縮幾何；這在「類別近似單模態」時有效，但在多模態類內分佈時，**等權正樣本平均**會把跨子模態的關係也當成應該強拉近的訊號。RASCAL 的想法是：

* **不是減少對齊**，而是把對齊目標從「所有同類平均」改成「對訓練動態穩定的同類關係優先」；
* 所以它仍是 contrastive，但對齊的是**穩定結構**而不是**標籤下的全連通假設**。
  ([Proceedings of Machine Learning Research][1])

---

## 方法論（嚴格按你指定格式）

### **Signal → Construction → Consistency → Realization/Integration**

### 1) Signal

**免費訊號：feature neighborhood stability（類內鄰域排序穩定度）**

對每個樣本 (i)，在當前 step/epoch 的投影特徵 (z_i^t)（已 L2 normalize）下，觀察同類樣本集合 (P(i)) 的相似度排序；再和一個 **detached cache**（上次看到該樣本時的 embedding，記為 (\bar z)）下的排序比較。

* 當前同類排序：(\rho_t(i,j))
* cache 同類排序：(\bar \rho(i,j))

這個排序差距就是「該正樣本關係是否穩定」的免費訊號。

---

### 2) Construction

**把 signal 轉成可對齊的結構目標：anchor-conditioned 非對稱 rank graph (W)**

對每個 anchor (i) 和 positive (j\in P(i))，定義正規化 rank drift：
[
\delta_{ij}
===========

\frac{|\rho_t(i,j)-\bar\rho(i,j)|}{|P(i)|-1}
\in [0,1].
]

定義穩定度權重（**無額外超參數**）：
[
w_{ij}
======

\frac{1-\delta_{ij}}{\sum_{p\in P(i)} (1-\delta_{ip})},
\qquad j\in P(i).
]

* 每一列 (w_i) 是一個機率分布（row-stochastic）
* 這形成一個 **非對稱 graph / rank target**（因為 (w_{ij}\neq w_{ji}) 一般成立）

> 直觀：同類中「排序關係穩定」的正樣本，應該被更強地拉近；不穩定者先別強拉。

---

### 3) Consistency

**用單一 loss 讓對比機率與穩定 rank 結構一致**

令
[
q_i(a)=
\frac{\exp(z_i^\top z_a/\tau)}
{\sum_{b\in A(i)}\exp(z_i^\top z_b/\tau)},
\quad a\in A(i),
]
其中 (A(i)) 是 batch 中除 anchor 外的全部樣本（含正負）。

定義 **RASCAL loss**（取代 SupCon Eq.2 的等權平均）：
[
\mathcal L_{\text{RASCAL}}
==========================

\sum_i \mathcal L_i,
\qquad
\mathcal L_i
============

-\sum_{p\in P(i)} w_{ip}\log q_i(p).
]

這裡沒有額外 regularizer、沒有 (\lambda)；整個 pretrain objective 就是這一個 loss。

---

### 4) Realization / Integration

**插拔式整合到 SupCon pipeline（兩階段訓練流程不變）**

SupCon 原本就是 encoder + projection head 的 contrastive pretraining，再接 linear probe / fine-tune；RASCAL 只是在 pretraining 階段把 SupCon 的等權正樣本項替換成上面的加權版本。投影頭一樣可丟棄、推論參數量不變。 

* **新增內容**：只是一個 loss（RASCAL）
* **額外資源**：一個 detached embedding cache（sample-id → last embedding）
* **參數量增加**：0
* **推論開銷增加**：0

---

## 數學理論推演與證明（可寫進方法章 / appendix）

### 命題 1：RASCAL 是「正樣本總量 + 結構分布對齊」的分解

定義正樣本總機率
[
Q_i^+ = \sum_{p\in P(i)} q_i(p),
]
以及在正樣本集合上的條件分布
[
\hat q_i(p)=\frac{q_i(p)}{Q_i^+}.
]

則有：
[
\mathcal L_i
============

-\log Q_i^+

* H(w_i)
* \mathrm{KL}(w_i|\hat q_i).
  ]

**證明（推演）**
[
\mathcal L_i
= -\sum_{p} w_{ip}\log q_i(p)
= -\sum_p w_{ip}\log\big(Q_i^+\hat q_i(p)\big)
]
[
= -\log Q_i^+ - \sum_p w_{ip}\log \hat q_i(p)
= -\log Q_i^+ + H(w_i)+\mathrm{KL}(w_i|\hat q_i).
]
證畢。

**意義**：RASCAL 不是單純「縮小正樣本距離」，而是同時要求

1. 正樣本總量要大（contrastive alignment）
2. 正樣本內部機率分配要匹配穩定 rank 結構（structure consistency）

---

### 命題 2：RASCAL 的梯度是「全體 softmax 重心」對「穩定正樣本重心」的差

設 (s_{ia}=z_i^\top z_a/\tau)，且 (|z|_2=1)。則
[
\frac{\partial \mathcal L_i}{\partial z_i}
==========================================

\frac{1}{\tau}
\left(
\sum_{a\in A(i)} q_i(a) z_a
---------------------------

\sum_{p\in P(i)} w_{ip} z_p
\right).
]

**證明（推演）**
[
\mathcal L_i = -\sum_p w_{ip}s_{ip} + \log\sum_{a}e^{s_{ia}}
]
對 (z_i) 微分即可得上式（(w_{ip}) 對當前 forward 視為 stop-grad 結構目標）。證畢。

**意義**：SupCon 的等權正樣本重心被替換成「穩定結構重心」，可避免跨子模態 positives 在早期被過度牽引。

---

### 命題 3：當類內排序完全穩定且等差（或 drift 常數）時，RASCAL 退化為 SupCon

若對某 anchor (i)，所有 positive 的 rank drift 相同（即 (\delta_{ij}=c_i) 對所有 (j\in P(i))），則
[
w_{ij} = \frac{1}{|P(i)|},
]
因此 (\mathcal L_i) 就是 SupCon Eq.2 的 per-anchor 形式。 

**意義**：RASCAL 在「類內真的是單模態 / 穩定全連通」時不會亂改，和 SupCon 一致；只有在存在結構 mismatch 時才發揮作用。

---

### 命題 4：相對 SupCon 的梯度擾動有界（穩定性保證）

令 (u_i) 為均勻正樣本權重，(g_i^{\text{SupCon}}, g_i^{\text{RASCAL}}) 分別為 anchor 梯度。則
[
|g_i^{\text{RASCAL}} - g_i^{\text{SupCon}}|_2
\le
\frac{1}{\tau}|w_i-u_i|_1.
]

因為 (|z_p|_2=1)，由三角不等式直接得證。

**意義**：RASCAL 是對 SupCon 的**受控偏移**；當 rank drift 小、(w_i) 近似均勻時，訓練動態幾乎不變。

---

## 與現有研究之區別

1. **對 SupCon 的區別**：SupCon 用等權正樣本平均；RASCAL 用 **rank-stability induced weights**，仍然是單一 contrastive loss，不加 auxiliary term。 
2. **對 hard positive / hard negative 觀點的區別**：SupCon 的分析強調 hard positives/negatives 的作用；RASCAL 不看「當下難度」，而看「跨訓練步的結構穩定性」，更偏向結構保真。 
3. **對 neural collapse / class collapse 文獻的區別**：既有理論多在分析塌縮幾何與最優解性質；RASCAL 是在訓練目標層面，用免費訊號把「必要塌縮」改成「結構感知塌縮」。 ([Proceedings of Machine Learning Research][2])
4. **對長尾方法的區別**：很多方法靠 re-weighting / re-sampling / expert routing；RASCAL 不引入多專家或 class-frequency 額外損失，而是在 SupCon 內部直接修正類內 attraction 假設。長尾設定只作為驗證場景。 ([openaccess.thecvf.com][3])

---

## 預計使用 dataset（依研究問題分層）

### A. 平衡分類（看純結構效果）

* **CIFAR-100**：有 fine/coarse labels，適合分析類內/類間結構與表示幾何。 ([cs.toronto.edu][4])
* **ImageNet (ILSVRC)**：大規模主流分類 benchmark。 ([arXiv][5])

### B. 細粒度分類（看多模態/子結構保留）

* **CUB-200-2011**：有 part locations 與 attributes，可做結構保留診斷。 ([vision.caltech.edu][6])
* **Stanford Cars**：細粒度車型辨識，類內外觀差異微妙。 
* **FGVC-Aircraft**：有層級標籤（variant/family 等），很適合驗證「排序穩定度是否對齊階層結構」。 ([robots.ox.ac.uk][7])

### C. 長尾/真實分布（看 mismatch 最明顯場景）

* **CIFAR-100-LT**（人工長尾，依常見 long-tail protocol）與 **ImageNet-LT**（OLTR benchmark） ([openaccess.thecvf.com][8])
* **iNaturalist 2018**：官方描述即強調 fine-grained + high class imbalance + longer tail。 ([Google Sites][9])

### D. 穩健性評估

* **ImageNet-C**：標準 common corruption benchmark（也是 SupCon 論文會報告的 robustness 指標之一）。 ([arXiv][10])

---

## Experiment 設計（可直接當實驗章骨架）

### 1) 訓練 protocol（公平比較）

* 完全沿用 SupCon 的兩階段流程：

  1. contrastive pretraining（只把 SupCon loss 換成 RASCAL loss）
  2. linear probe / fine-tune
* Augmentation、batch size、temperature (\tau)、backbone 全部與 SupCon baseline 對齊。 

---

### 2) Baselines

* **Cross-Entropy (CE)**
* **SupCon (Eq.2, official form)**
* （可選）只做「當前 similarity 加權」的 naive variant（不是主方法，只作 ablation 對照）

  * 用來證明「穩定度」比「當下相似度大小」更關鍵

---

### 3) 評估指標

#### 主指標

* Top-1 / Top-5 accuracy
* 長尾 setting：Many / Medium / Few-shot 分組 accuracy（依 ImageNet-LT 慣例） ([openaccess.thecvf.com][11])
* ImageNet-C：mCE 或 corruption-wise error（robustness） ([arXiv][10])

#### 結構診斷指標（本論文亮點）

* **Intra-class rank stability**（訓練過程曲線）
* **Within-class geodesic variance**（看是否過度塌縮）
* **Subclass retention proxy**：

  * CUB：用 attributes/parts 做同類內局部一致性評估 ([vision.caltech.edu][6])
  * FGVC-Aircraft：用 hierarchy level 做檢驗（variant vs family） ([robots.ox.ac.uk][7])

---

### 4) 關鍵 ablation（不違反單一貢獻）

> 只對 **RASCAL loss** 做剖析，不新增其他模組

* **w 的來源**：

  * uniform（即 SupCon）
  * current-only rank（無穩定度）
  * **rank-agreement（RASCAL）**
* **cache 更新策略**：last-seen vs epoch-end refresh（都不引入參數）
* **對 batch size 的敏感度**：檢查小 batch 下是否仍有效
* **對 label noise 的敏感度**：因排序穩定度應可抑制不穩定 positives（可作附錄）

---

### 5) 預期結果與假設驗證

* 在細粒度與長尾資料上，RASCAL 相對 SupCon 提升更明顯（因 mismatch 更強）
* 在平衡資料上不應明顯退步（由命題 3 支持：可退化回 SupCon）
* ImageNet-C / corruption robustness 可能維持或提升（結構一致性提升）
* 表示幾何上：RASCAL 會比 SupCon 保留更高類內局部結構，但不犧牲類間分離

---

如果你要，我可以下一步直接幫你把這題再展成一版 **paper proposal（摘要 + related work + method + theorem + experiment schedule）**，或把數學部分整理成更像論文 Appendix 的寫法。

[1]: https://proceedings.mlr.press/v119/wang20k/wang20k.pdf " Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere "
[2]: https://proceedings.mlr.press/v139/graf21a.html "Dissecting Supervised Contrastive Learning"
[3]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf "Class-Balanced Loss Based on Effective Number of Samples"
[4]: https://www.cs.toronto.edu/~kriz/cifar.html "CIFAR-10 and CIFAR-100 datasets"
[5]: https://arxiv.org/abs/1409.0575 "[1409.0575] ImageNet Large Scale Visual Recognition Challenge"
[6]: https://www.vision.caltech.edu/datasets/cub_200_2011/ "Perona Lab - CUB-200-2011"
[7]: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ "FGVC-Aircraft"
[8]: https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html "CVPR 2019 Open Access Repository"
[9]: https://sites.google.com/view/fgvc5/competitions/inaturalist "FGVC5 - iNaturalist"
[10]: https://arxiv.org/abs/1903.12261 "[1903.12261] Benchmarking Neural Network Robustness to Common Corruptions and Perturbations"
[11]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.pdf "Large-Scale Long-Tailed Recognition in an Open World"
