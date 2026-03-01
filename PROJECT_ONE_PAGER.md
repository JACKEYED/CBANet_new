# CBANet 项目一页纸导图

> 目标：帮助你在 5–10 分钟内建立“这个项目怎么 work”的整体认知。

---

## 1) 模块关系图（Module Relationship）

```mermaid
flowchart TD
    A[输入图像 x: Bx3xHxW] --> B[Encoder4: ImageCompressor_slimmable]

    B --> C1[AQL/IAQL 分支 idx=1]
    B --> C2[AQL/IAQL 分支 idx=2]
    B --> C3[AQL/IAQL 分支 idx=3]
    B --> C4[基础分支 idx=4 (无 AQL)]

    C1 --> D[latent feature]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E1[Decoder1]
    D --> E2[Decoder2]
    D --> E3[Decoder3]

    E1 --> F1[rec1]
    E1 --> G1[gate21]
    E1 --> G2[gate31]
    E2 --> G3[gate32]

    G1 --> F2[rec2 = gate21(rec1)+rec2_raw]
    G2 --> F3[part1]
    G3 --> F3
    E3 --> F3
    F3 --> F4[rec3 = gate31(rec1)+gate32(rec2_raw)+rec3_raw]

    F1 --> L[RD Loss 聚合]
    F2 --> L
    F4 --> L

    L --> O[反向传播更新 Encoder4/Decoder/Gate/AQL/IAQL]
    O --> S[sync_encoders: Encoder1/2/3 <- Encoder4]
```

### 核心理解
- 训练时只显式前向 `Encoder4`，但每个 epoch 后把权重同步到 `Encoder1/2/3`，以便测试脚本能按 checkpoint 字段完整加载。  
- 解码端是“逐级叠加”结构：
  - 一档：`rec1`
  - 二档：`gate21(rec1) + rec2_raw`
  - 三档：`gate31(rec1) + gate32(rec2_raw) + rec3_raw`

---

## 2) Tensor 尺寸流（Tensor Shape Flow）

设输入为 `x ∈ R^{B×3×H×W}`，且 `H,W` 可被 128 整除（测试集会裁齐）。

### 2.1 编码器与超先验主链路
1. `feature = Encoder(x)`：
   - 形状约为 `B × M × H/16 × W/16`（代码中量化噪声 feature 也按该尺度构造）。
2. `z = priorEncoder(feature or prior_feature)`：
   - 形状约为 `B × N × H/64 × W/64`（代码中量化噪声 z 按该尺度构造）。
3. `recon_sigma = priorDecoder(compressed_z)`：
   - 与 feature 对齐的概率参数张量，用于估计 feature 熵模型。
4. `compressed_feature_renorm`：
   - 训练：`feature + U(-0.5,0.5)` 近似量化；
   - 测试：`round(feature)`。

### 2.2 解码融合链路
- `rec1 = Decoder1(feature)`
- `rec2 = gate21(rec1) + Decoder2(feature)`
- `rec3 = gate31(rec1) + gate32(Decoder2(feature)) + Decoder3(feature)`

> 直观上：更高复杂度/码率分支会复用更低分支的信息，通过 gate 学习“复用比例”。

---

## 3) Loss 分解（Rate–Distortion）

## 3.1 单个输出的 RD Loss
对任意重建图 `recon`：

\[
\mathcal{L}_{RD}(recon, x; \lambda) = \lambda\cdot \mathrm{MSE}(recon,x) + \mathrm{bpp}
\]

其中：
- `MSE = mean((recon - x)^2)`
- `bpp = bpp_feature + bpp_z`
  - `bpp_feature`：基于 `Laplace(0, sigma)` 的特征熵估计；
  - `bpp_z`：基于 `BitEstimator` 的超先验熵估计。

## 3.2 一次训练 step 的总损失
- 共有 4 个质量分支（idx=1,2,3 用 AQL；idx=4 不用 AQL），每个分支都产生 3 个宽度输出（`rec1/rec2/rec3`）。
- 总计 12 个 RD loss，最终取平均：

\[
\mathcal{L}_{step} = \frac{1}{12}\sum_{i=1}^{4}\sum_{w=1}^{3}\mathcal{L}_{RD}^{(i,w)}
\]

其中各分支使用不同的 \(\lambda_i\)（由配置 `train_lambda1..4` 给出）。

---

## 4) 训练/评估最小闭环（你可直接照跑）

### 4.1 训练
```bash
python train.py \
  --config config/OneEncoderPruner.json \
  --train_data_dir PATH_TO_TRAIN_IMAGES \
  --save_dir checkpoints/cbanet_train \
  --epochs 30
```

### 4.2 快速 smoke run（建议先通路）
```bash
python train.py \
  --config config/OneEncoderPruner.json \
  --train_data_dir PATH_TO_TRAIN_IMAGES \
  --save_dir checkpoints/cbanet_train_smoke \
  --epochs 1 \
  --steps_per_epoch 5
```

### 4.3 测试
```bash
python test.py --config config/OneEncoderPruner.json -p checkpoints/cbanet_train/final_model.pth.tar
```

---

## 5) 快速记忆卡（30 秒复述）

- **一个编码器主干**（训练时前向用 `Encoder4`）+ **三个解码宽度分支**（Decoder1/2/3）。
- **Gate** 做跨宽度输出融合；**AQL/IAQL** 做自适应量化/反量化调节。
- 每步训练覆盖 **4 个质量档 × 3 个宽度输出 = 12 个 RD loss** 并求平均。
- bpp 由 feature 熵模型 + z 熵模型两部分组成。
