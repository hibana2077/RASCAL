可以，下面給你一份**可直接動手改 SupContrast 的實作引導**（以你前面那個 **RASCAL loss** 為目標）。我會用 **HobbitLong/SupContrast** 當 baseline，因為它的 `SupConLoss` 介面、`main_supcon.py` 訓練流程都很清楚，且官方 README 已明確說明 `features` 形狀、`labels` 用法與預訓練/線性評估流程。([GitHub][1])

---

## 0) 先對齊 baseline（你要改哪裡）

SupCon 論文的 Eq. (2) 是「**正樣本平均在 log 外面**」的版本（也就是常用的 SupCon 形式），並定義了 batch 內同類樣本集合 (P(i))。你的 RASCAL 就是把這個「等權平均」改成「由 rank stability 產生的權重平均」。

在 SupContrast repo 裡：

* `SupConLoss` 接收 `features: [bsz, n_views, f_dim]` 與 `labels: [bsz]`，`labels=None` 時會退化成 SimCLR。([GitHub][1])
* `main_supcon.py` 的訓練流程是：

  1. 兩個 augmentation 視圖先串起來 `images = cat([v1, v2])`
  2. 過 model 得到 features
  3. 再切回 `[bsz, 2, d]`
  4. 丟進 `criterion(features, labels)`。([GitHub][2])

所以你要改的核心只有兩個點：

1. **資料 loader 要多回傳 sample index**（給 cache 用）
2. **把 `SupConLoss` 換成 `RASCALLoss`**（單一 loss，無額外 λ）

---

## 1) 實作總體設計（最小改動版）

### 你要新增的唯一方法模組

* `losses_rascal.py` 內一個 `RASCALLoss(nn.Module)`

### 你要保留的東西

* Backbone / projection head 不動
* Augmentation 不動
* Pretrain + linear eval 流程不動（SupCon 的兩階段流程照走）

### 你要額外加的狀態（不是參數）

* 一個 **feature cache**（sample-id → 上次看到的 embedding）
* 一個 **valid mask**（哪個 sample 已經有 cache）

`register_buffer` 很適合放這種東西，因為它是 module state、不是可訓練參數，而且可以跟著 `.cuda()` 一起移動；也可設定 `persistent=False` 不寫進 `state_dict`。PyTorch 文件有明確說明。([PyTorch Docs][3])

---

## 2) 先改 DataLoader：讓它回傳 sample index

SupContrast 原本 train loop 是 `(images, labels)`，你要改成 `(images, labels, index)`。([GitHub][2])

### 做法：包一層 dataset wrapper

```python
# datasets_indexed.py
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]   # x 通常是 TwoCropTransform 後的 [view1, view2]
        return x, y, idx
```

### 在 `main_supcon.py` 的 `set_loader` 改一下

原本：

```python
train_dataset = datasets.CIFAR100(..., transform=TwoCropTransform(...))
```

改成：

```python
base_dataset = datasets.CIFAR100(..., transform=TwoCropTransform(...))
train_dataset = IndexedDataset(base_dataset)
```

> 如果你用自訂資料夾，SupContrast README 也有寫 `ImageFolder` 的資料結構慣例（`./path/class_name/xxx.png`）。([GitHub][1])

---

## 3) RASCAL loss 的實作骨架（核心）

下面是**建議的第一版**：先做到「清楚、可跑、容易 debug」，之後再優化向量化。

### 設計重點

* 仍然沿用 SupCon 的 logits / log_prob 計算
* **只改正樣本聚合方式**：

  * SupCon：`mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs`
  * RASCAL：`weighted_log_prob_pos = (W * log_prob).sum(1)`，其中 `W` 是你用 rank drift 建出的 row-stochastic 權重

### `losses_rascal.py`（可直接當起點）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RASCALLoss(nn.Module):
    """
    Rank-Agreement Supervised Contrastive Alignment Loss (RASCAL)
    單一 loss，取代 SupConLoss 中的等權正樣本平均。
    """
    def __init__(self, temperature=0.07, base_temperature=0.07,
                 num_samples=None, feat_dim=None, persistent_cache=False):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

        # cache: sample-level embedding cache (非參數)
        assert num_samples is not None and feat_dim is not None
        self.register_buffer(
            "cache_feat",
            torch.zeros(num_samples, feat_dim, dtype=torch.float32),
            persistent=persistent_cache
        )
        self.register_buffer(
            "cache_valid",
            torch.zeros(num_samples, dtype=torch.bool),
            persistent=persistent_cache
        )

    @torch.no_grad()
    def update_cache(self, sample_idx: torch.Tensor, sample_feat: torch.Tensor):
        """
        sample_idx: [bsz]
        sample_feat: [bsz, d] (detached, normalized)
        """
        self.cache_feat[sample_idx] = sample_feat
        self.cache_valid[sample_idx] = True

    def _build_rank_weights(self, contrast_feat_det, labels_rep, sample_idx_rep, logits_mask, pos_mask):
        """
        contrast_feat_det: [M, d] detached & normalized, M = bsz * n_views
        labels_rep:        [M]
        sample_idx_rep:    [M]
        logits_mask:       [M, M] self-contrast 已遮掉
        pos_mask:          [M, M] 同類正樣本 mask（含 self-sample 的另一視圖，但不含 self-view）
        回傳:
            W: [M, M]，只在 positives 位置有值，每列和為 1（若該列無正樣本則全零）
        """
        device = contrast_feat_det.device
        M = contrast_feat_det.size(0)

        # view-level current similarity（detach）
        sim_cur = contrast_feat_det @ contrast_feat_det.T  # [M, M]

        # view-level cached similarity（sample cache 映射到 view）
        cache_valid_rep = self.cache_valid[sample_idx_rep]        # [M]
        cache_feat_rep = self.cache_feat[sample_idx_rep]          # [M, d]
        sim_cache = cache_feat_rep @ cache_feat_rep.T             # [M, M]

        W = torch.zeros((M, M), device=device, dtype=contrast_feat_det.dtype)

        for r in range(M):
            pos_idx = torch.where(pos_mask[r] > 0)[0]  # positives of anchor r
            m = pos_idx.numel()
            if m == 0:
                continue

            # 若 anchor 或 positives 缺 cache，就退化成 uniform（避免前期 cache 太空）
            if (not cache_valid_rep[r]) or (not torch.all(cache_valid_rep[pos_idx])):
                W[r, pos_idx] = 1.0 / m
                continue

            # 取出 current / cached positive similarities
            cur_scores = sim_cur[r, pos_idx]
            cache_scores = sim_cache[r, pos_idx]

            # 轉成 ranking（0 是最相近）
            # argsort(argsort()) 寫法夠直觀；後續可再向量化優化
            rank_cur = torch.argsort(torch.argsort(-cur_scores))
            rank_cache = torch.argsort(torch.argsort(-cache_scores))

            if m == 1:
                drift = torch.zeros_like(cur_scores)
            else:
                drift = (rank_cur - rank_cache).abs().float() / (m - 1)

            # 無額外超參數：w = 1 - drift，再做 row normalization
            w = (1.0 - drift).clamp_min(0.0)
            s = w.sum()
            if s <= 1e-12:
                W[r, pos_idx] = 1.0 / m
            else:
                W[r, pos_idx] = w / s

        return W

    def forward(self, features, labels, sample_idx):
        """
        features:   [bsz, n_views, d]
        labels:     [bsz]
        sample_idx: [bsz]
        """
        if features.ndim < 3:
            raise ValueError("features must be [bsz, n_views, ...]")
        if features.ndim > 3:
            features = features.view(features.size(0), features.size(1), -1)

        bsz, n_views, d = features.shape
        device = features.device

        # 建議保險起見在 loss 內再 normalize 一次
        features = F.normalize(features, dim=-1)

        # ===== SupCon 標準展平（對齊官方實作）=====
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz*n_views, d]
        M = contrast_feature.size(0)

        # labels / sample_idx 展平成 view-level
        labels = labels.contiguous().view(-1, 1)                     # [bsz,1]
        mask_sample = torch.eq(labels, labels.T).float().to(device)  # [bsz,bsz]

        # SupCon 的 row/col 擴張方式
        pos_mask = mask_sample.repeat(n_views, n_views)              # [M,M]

        # self-contrast mask
        logits_mask = torch.ones((M, M), device=device)
        logits_mask.fill_diagonal_(0.)
        pos_mask = pos_mask * logits_mask

        # ===== logits / log_prob（跟 SupCon 一樣）=====
        logits = torch.div(contrast_feature @ contrast_feature.T, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # ===== RASCAL 權重 W =====
        labels_rep = labels.view(-1).repeat(n_views)      # [M]
        sample_idx_rep = sample_idx.view(-1).repeat(n_views).to(device)  # [M]

        W = self._build_rank_weights(
            contrast_feat_det=contrast_feature.detach(),
            labels_rep=labels_rep,
            sample_idx_rep=sample_idx_rep,
            logits_mask=logits_mask,
            pos_mask=pos_mask
        )

        # 單一 loss：用 W 取代 SupCon 的等權平均
        weighted_log_prob_pos = (W * log_prob).sum(1)
        loss = - (self.temperature / self.base_temperature) * weighted_log_prob_pos
        loss = loss.mean()

        # ===== 更新 cache（sample-level）=====
        # 用 sample-level feature 作為 cache（兩個 view 取平均）
        with torch.no_grad():
            sample_feat = features.mean(dim=1)            # [bsz, d]
            sample_feat = F.normalize(sample_feat, dim=-1)
            self.update_cache(sample_idx.to(device), sample_feat.detach())

        return loss
```

---

## 4) 在 `main_supcon.py` 怎麼接上去

SupContrast 原本 `train()` 迴圈會拿到 `(images, labels)`，你改成 `(images, labels, indices)`。然後把 `criterion(features, labels)` 改成 `criterion(features, labels, indices)`。訓練時 `images` 先 concat、features 再 split/stack 的流程照原本方式。這個流程在官方 `main_supcon.py` 就是這樣寫的。([GitHub][2])

### 你要改的地方（示意）

```python
# 原本
for idx, (images, labels) in enumerate(train_loader):
    ...
    loss = criterion(features, labels)

# 改成
for step, (images, labels, indices) in enumerate(train_loader):
    ...
    if torch.cuda.is_available():
        indices = indices.cuda(non_blocking=True)

    loss = criterion(features, labels, indices)
```

### 初始化 criterion（要知道 `num_samples` 和 `feat_dim`）

`num_samples = len(train_loader.dataset)`（wrapper 後仍可取長度）
`feat_dim` 要跟 projection head 輸出維度一致（你目前 backbone 設定是多少就填多少）。

---

## 5) 先做這 3 個 sanity check（非常重要）

### A. 退化檢查：RASCAL 是否能退回 SupCon

把 `_build_rank_weights(...)` 先暫時改成 uniform：

```python
W[r, pos_idx] = 1.0 / m
```

此時結果應該要和 SupCon 非常接近（數值差可能只來自你的 `1e-12` 與 normalize 細節）。

### B. 無正樣本行是否會 NaN

SupCon 原始實作特別處理了「某 anchor 在 batch 內沒有正樣本」時的除零 edge case。你的版本也要保證這種 row 的 loss 不出 NaN。官方 `losses.py` 有這個保護邏輯。([GitHub][4])

### C. 前幾個 epoch cache 很空時是否穩定

前期很多樣本 `cache_valid=False`，你已經設計成 fallback uniform；這樣 loss 應該會平順，不會一開始就亂跳。

---

## 6) DDP 版（建議你第二步再做）

SupContrast README 自己就提醒過：目前 repo 用的是 `DataParallel`，`syncBN` 在那個寫法下沒有效果；如果你要做比較正式的大實驗，建議改 DDP。([GitHub][1])

### 為什麼 RASCAL 在 DDP 要多一步

你要用「整個 global batch」算 positives/negatives（否則每卡 batch 太小、正樣本不足）。所以要把每張卡的：

* `features`
* `labels`
* `sample_idx`

都 `all_gather` 起來。PyTorch 的 `torch.distributed.all_gather` 就是把每個 rank 的 tensor 收集成 list。文件有明確定義與範例。([PyTorch Docs][5])

### DDP 實作要點

1. local forward 得到 `features_local`
2. `all_gather` 成 `features_global`
3. loss 用 `features_global / labels_global / idx_global` 算
4. cache 更新也用 gathered 的 sample features（讓每張卡 cache 一致）

> 先把單卡/單機版本跑通，再上 DDP，會省很多時間。

---

## 7) 建議的 debug 記錄（你之後寫論文也會用到）

你這題的亮點是「結構目標」，所以建議從第一天就記以下統計：

### 每 step / 每 epoch 記錄

* `loss`
* `avg_pos_per_anchor`（batch 內每列正樣本數）
* `cache_hit_rate`（有效 cache 比例）
* `rank_drift_mean`
* `rank_drift_std`
* `W_entropy`（每列權重熵，越低表示越偏向少數穩定 positives）

### 幾個你一定會想看的圖

1. **rank drift 分佈隨 epoch 變化**
2. **RASCAL vs SupCon 的 class-wise accuracy（尤其 long-tail few-shot）**
3. **類內 t-SNE / UMAP（看 sub-mode 是否被保留）**

---

## 8) 實作上的常見坑（我先幫你避掉）

### 坑 1：sample index 跟 augmentation view 對不起來

你 cache 是 sample-level，不是 view-level；所以 `sample_idx` 要對兩個 views 重複（`repeat(n_views)`），不能直接拿 batch row index 亂用。

### 坑 2：cache 不要進梯度圖

`cache_feat` 更新一定要在 `torch.no_grad()` 裡，而且 `_build_rank_weights` 用的 current feature 最好 `.detach()`。你的結構目標是 supervision signal，不該讓排名本身反向穿回去。

### 坑 3：排序 ties（同分）

cache 初期可能很多相似度接近，`argsort` ties 會有任意性。這通常不致命，但你可以：

* 先接受（最簡單）
* 或加一個極小 deterministic tie-break（例如依 column index 的微小偏移）
  先不要過度工程化。

### 坑 4：`feat_dim` 不一致

如果 projection head 輸出維度改了（例如不是 128），記得 `RASCALLoss(feat_dim=...)` 也要改，不然 cache shape 會炸。

---

## 9) 你可以照這個順序做（最快出結果）

### Phase 1（半天～1 天）

* [ ] `IndexedDataset`
* [ ] `RASCALLoss`（單卡版）
* [ ] CIFAR-10 / CIFAR-100 跑通（先小 batch 也行）
* [ ] sanity check：RASCAL→uniform ≈ SupCon

### Phase 2（1～2 天）

* [ ] log rank drift / cache hit / W entropy
* [ ] 做 ablation：`uniform` vs `current-rank-only` vs `rank-agreement`

### Phase 3（之後）

* [ ] DDP + all_gather
* [ ] 細粒度資料（CUB / Aircraft）
* [ ] 長尾設定（CIFAR-LT / ImageNet-LT）

---

## 10) 如果你要更穩的第一版，我建議這樣收斂

先做一個「**RASCAL-Lite**」版本（完全不改 batch 結構，只改 loss）：

* 用 SupContrast 原始 augmentation / backbone / optimizer
* `W` 用 row-loop 寫（慢一點沒關係）
* 不做 DDP
* 先在 CIFAR-100 跑 100 epoch 看趨勢

只要你先看到：

* loss 正常下降
* `rank_drift_mean` 逐步下降
* 比 SupCon 好一點點（哪怕 +0.2%）

這題就已經很有論文味了。

---

如果你要，我下一則可以直接給你一版**可貼進 SupContrast 的 patch 版程式碼**（按檔案分段：`datasets_indexed.py`、`losses_rascal.py`、`main_supcon.py` 修改片段），讓你幾乎直接複製就能跑。

[1]: https://github.com/HobbitLong/SupContrast "GitHub - HobbitLong/SupContrast: PyTorch implementation of \"Supervised Contrastive Learning\"  (and SimCLR incidentally)"
[2]: https://raw.githubusercontent.com/HobbitLong/SupContrast/master/main_supcon.py "raw.githubusercontent.com"
[3]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html "Module — PyTorch 2.10 documentation"
[4]: https://raw.githubusercontent.com/HobbitLong/SupContrast/master/losses.py "raw.githubusercontent.com"
[5]: https://docs.pytorch.org/docs/stable/distributed.html "Distributed communication package - torch.distributed — PyTorch 2.10 documentation"
