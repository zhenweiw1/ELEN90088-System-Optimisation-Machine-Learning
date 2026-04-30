# SOML 2026 Exercise 2 — 知识点完全笔记

> 写给"什么都不会"的同学。每节先讲**直觉**，再写**公式**，最后给**代码模板**。
> 公式全部用纯文本/Unicode 写，任何 Markdown 预览都能看。
>
> 涉及四大主题：
> 1. 贝叶斯推断 / MLE / MAP / 在线学习
> 2. K-means 与高斯混合模型 (GMM)
> 3. 神经网络 (DNN, PyTorch)
> 4. 卷积变分自编码器 (Convolutional VAE)

---

## 0. 数学基础速查

### 0.1 概率密度 (PDF) vs 概率质量 (PMF)
- **离散**分布用 PMF：`P(X = x)`。例：抛硬币 `P(X = 1) = θ`。
- **连续**分布用 PDF：`p(x)`，本身不是概率，要积分才是。

### 0.2 伯努利分布 (Bernoulli)
- 一次抛硬币的模型：`X ∈ {0, 1}`，`P(X=1) = θ`。
- PMF 一行写：

```
p(x | θ) = θ^x · (1 − θ)^(1 − x)
```

### 0.3 Beta 分布
- 定义在 `[0, 1]` 上，常用作"概率参数 θ 的先验"。
- PDF：

```
p(θ) = (1 / B(a, b)) · θ^(a − 1) · (1 − θ)^(b − 1)
```

- 形状直觉：
  - `a = b = 1`：均匀分布（什么都不知道）。
  - `a > b`：偏向 1（更可能正面）。
  - `a < b`：偏向 0（更可能反面）。
  - `a, b` 都很大：尖峰，先验信念很强。
- **共轭性**：Beta 是 Bernoulli 的共轭先验，先验 Beta、后验也是 Beta。

### 0.4 高斯（正态）分布
- 一维：`N(μ, σ²)`，PDF：

```
p(x) = 1 / (sqrt(2π) · σ) · exp( −(x − μ)² / (2σ²) )
```

- d 维对角高斯：`N(μ, diag(σ²))`，每一维独立。
- 标准正态：`μ = 0, σ = 1`。

### 0.5 贝叶斯定理

```
p(θ | x) = p(x | θ) · p(θ) / p(x)
         ∝ p(x | θ) · p(θ)
```

- `p(θ)`：**先验**，看到数据前的信念。
- `p(x | θ)`：**似然**，给定 θ 数据的概率。
- `p(θ | x)`：**后验**，看到数据后的更新信念。
- `p(x)` 是归一化常数，往往可忽略。

### 0.6 KL 散度
衡量两个分布的"距离"（不对称）：

```
KL(q ‖ p) = ∫ q(z) · log( q(z) / p(z) ) dz
```

- 始终 ≥ 0，仅当 `q = p` 时为 0。
- VAE 里就是用它把后验拉向先验。

---

## 1. Question 1：贝叶斯推断、MLE、MAP、在线学习

### 1.1 整体故事
你捡到一枚不知是否公平的硬币。
- 用 **Beta 先验** 表达初始直觉。
- 抛硬币更新信念 → 用 Bayes 定理算**后验**。
- 想要一个"最佳点估计"：MLE 或 MAP。
- 在线预测：每一步用之前的数据估计 θ，再做决定。

### 1.2 后验推导（Part A）
**结论**：抛 n 次后，正面次数 `n_H = Σ x_i`，则

```
θ | x_1, ..., x_n  ~  Beta(n_H + a,  n − n_H + b)
```

**直觉**：先验中的 a, b 就像"假装提前观察过 a 次正面、b 次反面"。

**推导套路**（似然 × 先验，约掉常数）：

```
p(θ | x_1:n) ∝ [ Π_i θ^{x_i} (1 − θ)^{1 − x_i} ] · θ^{a − 1} (1 − θ)^{b − 1}
            =  θ^{n_H + a − 1} · (1 − θ)^{n − n_H + b − 1}
```

认出来这正是 Beta 的核 → 共轭。

### 1.3 MLE：最大似然估计
- 只用数据，不用先验。
- 对硬币的对数似然：

```
ℓ(θ) = n_H · log θ + (n − n_H) · log(1 − θ)
```

- 求导 = 0：

```
θ_MLE = n_H / n        （正面次数 / 总次数）
```

### 1.4 MAP：最大后验估计
- 同时用先验 + 数据。最大化后验密度：

```
θ_MAP = (n_H + a − 1) / (n + a + b − 2)        （要求 a, b ≥ 1）
```

- **直觉**：把先验的 `a−1` 次"伪正面"和 `b−1` 次"伪反面"加进去。
- `n → ∞` 时 MAP → MLE，先验影响被数据淹没。

### 1.5 实现要点（Part C）
- 每一步累加 `n_H`，直接用上面的公式更新。
- 同图画 MLE、MAP，再加一条真实 θ 的水平虚线作对照。

```python
n_H = 0
theta_mle, theta_map = [], []
for n_idx, x in enumerate(tosses, start=1):
    n_H += x
    theta_mle.append(n_H / n_idx)
    theta_map.append((n_H + a - 1) / (n_idx + a + b - 2))
```

### 1.6 在线学习与正则化遗憾 (Part D)
**问题**：每一步只能看过去，预测当前结果，正确 +1、错误 −1。

#### 最优策略（已知 θ）
- 期望奖励：预测 1 得 `2θ − 1`；预测 0 得 `1 − 2θ`。
- 所以最优规则就是**阈值规则**：

```
x*_i = 1   if  θ ≥ 0.5
x*_i = 0   otherwise
```

- 题目里 `θ = 0.3555 < 0.5` → 最优策略 = **总是预测 0**。

#### MLE 在线策略（不知 θ）
- 用前 i−1 次的频率作为 θ̂，再用阈值 0.5 决策；第一次默认预测 1。

#### 归一化遗憾 (regret)

```
R_n = (1/n) · ( Σ r*_i  −  Σ r_i )
```

衡量"不知道真实参数"造成的平均损失。

#### 关键观察
- n 小：估计误差大，`R_n` 大。
- `n → ∞`：MLE 是一致的（θ̂ → θ），错误次数趋于稳定，`R_n → 0`。
- 这就是"渐近最优"：长期看就跟知道真值一样。

---

## 2. Question 2：聚类 (K-means & GMM)

### 2.1 监督 vs 无监督
- **监督学习**：有标签 `(x, y)`，做分类 / 回归。
- **无监督学习**：只有 `x`，从结构本身找模式；聚类是典型例子。

### 2.2 K-means

#### 算法（重复直到收敛）
1. **分配步**：每个点归到最近的中心点。
2. **更新步**：每个簇的中心 = 簇内点均值。

#### 目标函数 (within-cluster SSE)

```
J = Σ_{k=1..K}  Σ_{x ∈ C_k}  ‖x − μ_k‖²
```

- 单调下降，必收敛，但只到**局部极小**。

#### 初始化与超参
- `init='random'`：随机选中心点；可能收敛到差解。
- `k-means++`（默认）：聪明地分散初始中心，更稳。
- `n_init`：跑多次取最好的，建议 ≥ 10。
- K（簇数）需人工指定 → 用"肘部法"或外部指标判断。

#### 局限
- 假设簇是**凸的、各向同性的**（圆形）。
- 对 *moons / circles* 这类弯曲形状不灵。
- 对尺度敏感 → 通常先标准化。
- **硬分配**，没有不确定性。

#### sklearn 模板
```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
labels  = km.labels_
centers = km.cluster_centers_
```

### 2.3 高斯混合模型 (GMM)

#### 模型
认为数据由 K 个高斯混合而成：

```
p(x) = Σ_{k=1..K}  π_k · N(x; μ_k, Σ_k),   Σ_k π_k = 1
```

#### 训练：EM 算法
- **E 步**：算每个点对每个簇的*责任度* `γ_{ik} = p(z = k | x_i)`。
- **M 步**：用加权样本更新 `π_k, μ_k, Σ_k`。
- 与 K-means 关系：K-means 是 GMM 的"硬分配 + 球形协方差"特例。

#### 软分配
`predict_proba(X)` 返回每点对每簇的概率：

```python
import pandas as pd
df = pd.DataFrame(gmm.predict_proba(X)[:5], columns=['P(c0)', 'P(c1)'])
```

#### 选择簇数 K：BIC / AIC
- 对数似然总会随 K 上升 → 不能直接最大化它。
- 加上复杂度惩罚：

```
AIC = 2k − 2 · ln L
BIC = k · ln n − 2 · ln L     （惩罚更重，倾向于更小的 K）
```

- 选使 BIC（或 AIC）最小的 K。

#### GMM 也搞不定 moons
- 高斯本身是椭圆形的，moons 是弯月形 → 单个高斯无法刻画。
- 但 GMM 至少给出**概率**，比 K-means 信息更丰富。

---

## 3. Question 3：神经网络 (DNN)

### 3.1 总体结构

```
h^(ℓ) = σ( W^(ℓ) · h^(ℓ−1) + b^(ℓ) )
```

- 输入层 → 多个隐藏层（线性 + 非线性激活）→ 输出层。
- 二分类：最后一层 1 个节点，配合 sigmoid 输出概率。

### 3.2 激活函数
- **ReLU**：`max(0, x)`。默认选择，计算简单、缓解梯度消失。
- **tanh**：输出 `[−1, 1]`，对称；深网络易梯度消失。
- **Sigmoid**：`1 / (1 + e^{−x})`。常用作二分类输出层。

### 3.3 损失函数
- 回归：**MSE**。
- 二分类：**Binary Cross Entropy**：

```
L = −[ y · log p̂  +  (1 − y) · log(1 − p̂) ]
```

  PyTorch 推荐 `BCEWithLogitsLoss`：内部含 sigmoid + log，数值更稳。
- 多分类：**CrossEntropyLoss**（自带 softmax）。

### 3.4 优化器
- **SGD**：基础，需小心调 lr。
- **Adam**：自适应学习率，默认首选，`lr=1e-3` 一般稳。
- 学习率太大 → 不收敛/震荡；太小 → 收敛极慢。

### 3.5 反向传播 (backprop)
- 链式法则按层反向传梯度 `∂L / ∂W^(ℓ)`。
- PyTorch 自动算（`loss.backward()`）。
- 训练循环固定三步：

```python
optimizer.zero_grad()              # 1. 清梯度
loss = criterion(model(x), y)      # 2. 前向 + 算损失
loss.backward()                    # 3. 反向传播
optimizer.step()                   # 4. 更新参数
```

### 3.6 PyTorch DNN 模板
```python
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim, layer_list, activation=nn.ReLU):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for w in layer_list:
            self.layers.append(nn.Linear(prev, w))
            prev = w
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:    # 最后一层不加激活
                x = self.act(x)
        return x   # 返回 logits
```

### 3.7 训练 / 测试集
- `train_test_split(X, y)`：默认 75% / 25%。
- 训练集更新参数；**测试集只在最后评估**，不能拿来调参。

### 3.8 过拟合 vs 欠拟合
- **欠拟合**：模型太弱，训练 / 测试都差。
- **过拟合**：训练好、测试差 → 模型记住了训练点的噪声。
- 应对：早停、L2 正则 (weight decay)、Dropout、数据增强、增加数据量。
- **训练 acc 与测试 acc 差距大 = 过拟合的信号**。

### 3.9 epoch 实验直觉
- 前期：train / test loss 一起下降。
- 后期：train loss 仍降，但 test loss **不降甚至上升**。最佳模型常在中间某个 epoch。

### 3.10 更高级的网络
- **CNN**（卷积网络）：图像首选。
- **RNN / LSTM / Transformer**：序列数据。
- **ResNet**：跳连接训练超深网络。
- **BatchNorm / LayerNorm / Dropout**：帮助优化与正则化。

---

## 4. Question 5：变分自编码器 VAE

### 4.1 自编码器 (AE) 回顾
- 编码器 `E` 把 x 压成 z；解码器 `D` 重建 `x̂ = D(E(x))`。
- 目标：最小化重建误差 `‖x − x̂‖²`。
- 问题：潜空间没结构，随机采 z 解码出的样本通常是垃圾。

### 4.2 VAE 的关键想法
不让 z 是固定点，而是让它服从一个分布 `q(z|x)`，并把这分布拉向简单先验 `p(z) = N(0, I)`。
- 编码器输出**均值 μ 和对数方差 log σ²**。
- 采样 `z ~ N(μ, σ²)`。
- 解码器从 z 重建 `x̂`。

### 4.3 ELBO 与训练目标
最大化证据下界 (ELBO)：

```
L_ELBO = E_{q(z|x)}[ log p(x|z) ]  −  KL( q(z|x) ‖ p(z) )
```

等价于最小化 **负 ELBO = 重建损失 + KL 散度**。

#### 重建损失
- 像素值在 `[0, 1]` → 用 MSE 或 BCE。
- 用 `reduction='sum'`，与 KL 单位一致（不要混 `mean`）。

#### KL（对角高斯 q vs 标准正态 p）

```
KL = −0.5 · Σ_{j=1..d} ( 1 + log σ_j² − μ_j² − σ_j² )
```

- 推导：写出两个高斯的 PDF，代入 `KL = ∫ q · log(q/p)`，逐项算。
- 维度独立 → 总 KL = 各维之和。

```python
def vae_loss(recon_x, x, mu, log_var):
    recon = F.mse_loss(recon_x, x, reduction='sum')
    kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + kl, recon, kl
```

### 4.4 为什么需要 Reparameterization Trick？
- 直接 `z = sample(N(μ, σ²))` 是**随机操作**，对 μ, σ 没有可导路径，反向传播不通。
- 解法：把随机性"挪出去"

```
z = μ + σ ⊙ ε,    ε ~ N(0, I)
```

- 这样 z 是 μ, σ 的可微函数，ε 与参数无关，正常 backprop。

#### Toy 例子
- 设 `μ = 2, σ = 1, ε = 0.5` → `z = 2.5`；损失 `L = (z − 5)² = 6.25`。
- `∂L/∂z = 2(z − 5) = −5`，于是
  - `∂L/∂μ = −5 · 1   = −5`
  - `∂L/∂σ = −5 · 0.5 = −2.5`

### 4.5 卷积层基础
- `Conv2d(in, out, kernel=3, stride=2, padding=1)`：把 H×W 缩到一半。
- `ConvTranspose2d`（反卷积/上采样）：把分辨率放大。
- CIFAR-10：3 通道、32×32 图像。
- 目标 stride=2 三次：32 → 16 → 8 → 4。

### 4.6 ConvVAE 架构（题目指定）

```
编码器:  Conv(3 → 16) + ReLU
        Conv(16 → 32) + ReLU
        Conv(32 → 64) + ReLU      # 输出 64×4×4 = 1024
        Linear(1024, latent_dim) → mu
        Linear(1024, latent_dim) → log_var

采样:    z = mu + exp(0.5 · log_var) · randn_like(...)

解码器:  Linear(latent_dim, 1024) + ReLU → reshape 64×4×4
        ConvT(64 → 32) + ReLU
        ConvT(32 → 16) + ReLU
        ConvT(16 → 3)  + Sigmoid    # 输出在 [0, 1]
```

### 4.7 数据预处理
- `transforms.ToTensor()` 把 0–255 缩到 `[0, 1]` → 与 Sigmoid 输出对应。
- 不要再做 `Normalize(mean, std)`，否则像素不在 `[0, 1]`，与 Sigmoid 不一致。

### 4.8 训练循环模板
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(epochs):
    model.train()
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, log_var = model(x)
        loss, _, _ = vae_loss(recon, x, mu, log_var)
        loss.backward()
        optimizer.step()
```

### 4.9 测试函数（仅 MSE）
```python
def test(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon, _, _ = model(x)
            total += F.mse_loss(recon, x, reduction='sum').item()
            n     += x.size(0)
    return total / n
```

### 4.10 压缩率 vs 重建误差
- 原图比特数：`32 × 32 × 3 × 8 = 24576` bits。
- 潜变量比特数：`d × 32` bits（float32）。
- 压缩率：`(32 · d) / 24576`。
- 现象：d 越大 → 重建 MSE 越小，但**收益递减**。
  - 小 d（如 16）：信息瓶颈紧，重建模糊。
  - 大 d（如 256）：几乎不损细节，但占空间。
- 这就是经典的 **率–失真 (rate-distortion) 曲线**。

---

## 5. 复习清单（考前 5 分钟）

| 主题 | 要记住的一句话 |
|---|---|
| Beta 先验 + Bernoulli 似然 | 后验是 Beta，参数加上正反次数 |
| MLE | `θ_MLE = n_H / n` |
| MAP | `θ_MAP = (n_H + a − 1) / (n + a + b − 2)` |
| 在线最优策略 | 阈值 0.5：哪边期望大就预测哪边 |
| 归一化遗憾 | 数据多了趋近 0 |
| K-means 缺点 | 只对凸/球形簇有效，硬分配 |
| GMM 优点 | 软分配 + 密度估计 |
| 选 K | 看 BIC / AIC 最小点 |
| 二分类损失 | `BCEWithLogitsLoss` |
| 训练循环 | zero_grad → forward → loss → backward → step |
| 过拟合信号 | 训练好测试差 |
| Reparam Trick | `z = μ + σ · ε`，让随机性可导 |
| VAE 损失 | 重建 + KL |
| KL（对角高斯 vs N(0, I)） | `−0.5 · Σ ( 1 + log σ² − μ² − σ² )` |
| 压缩率 vs 误差 | 维度越大重建越好，收益递减 |

祝顺利通过! 💪
