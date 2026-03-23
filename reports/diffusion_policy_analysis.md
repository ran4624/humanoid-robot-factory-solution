# Diffusion Policy 深度分析报告

**报告日期**: 2026-03-22  
**VLA 4周速成计划 - Week 2 Day 3**  
**论文**: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (Columbia University, 2023)

---

## 一、论文基本信息

| 属性 | 信息 |
|-----|------|
| **标题** | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion |
| **作者** | Cheng Chi*, Zhenjia Xu* et al. (Columbia University, Toyota Research Institute, MIT, Stanford) |
| **发表** | arXiv:2303.04137, RSS 2023 (扩展期刊版本 2024) |
| **代码** | https://diffusion-policy.cs.columbia.edu |
| **核心创新** | 将扩散模型应用于机器人视觉运动策略学习 |

---

## 二、核心问题与动机

### 2.1 传统方法的局限

机器人模仿学习通常被形式化为有监督回归任务：从观测映射到动作。但动作预测有以下独特挑战：

1. **多模态分布** (Multimodal): 同一场景可能有多个正确动作
2. **时序相关性** (Sequential correlation): 动作需要平滑连续
3. **高精度要求** (High precision): 机器人需要精确控制

### 2.2 现有方法的不足

| 方法类型 | 代表 | 局限 |
|---------|------|------|
| 显式策略 (Explicit) | GMM, 离散分类 | 难以表达多模态分布 |
| 隐式策略 (Implicit) | EBMs | 训练不稳定，需要负采样 |

### 2.3 Diffusion Policy的核心洞察

> **将机器人策略表示为条件去噪扩散过程**

通过继承扩散模型的强大生成能力，解决上述三个挑战。

---

## 三、算法实现细节

### 3.1 扩散模型基础 (DDPM)

**去噪过程** (Denoising Process):
```
x^{k-1} = α(x^k - γ·ε_θ(x^k, k) + N(0, σ²I))
```

其中:
- `x^k`: 第k次迭代时的带噪动作
- `ε_θ`: 噪声预测网络（核心参数）
- `α, γ, σ`: 噪声调度参数
- `N(0, σ²I)`: 高斯噪声

**梯度下降视角**:
```
x' = x - γ·∇E(x)
```
噪声预测网络 `ε_θ` 实际上预测的是能量函数 `E(x)` 的梯度场。

### 3.2 训练过程

**训练目标** (MSE Loss):
```
L = MSE(ε^k, ε_θ(O_t, A_t^0 + ε^k, k))
```

训练步骤:
1. 从数据集采样干净动作 `A_t^0`
2. 随机选择去噪迭代 `k`
3. 采样噪声 `ε^k` 并加到动作上
4. 噪声预测网络学习预测这个噪声

### 3.3 视觉运动策略的关键修改

#### 修改1: 闭环动作序列预测

**时间参数定义**:
- `T_o`: 观测时域 (observation horizon)
- `T_p`: 动作预测时域 (prediction horizon)  
- `T_a`: 动作执行时域 (execution horizon, T_a ≤ T_p)

**流程**:
```
At time t:
1. 输入: 最近 T_o 步观测 O_t
2. 预测: T_p 步动作序列 A_t
3. 执行: 仅执行前 T_a 步动作
4. 重复: 在 t+T_a 时刻重新规划
```

**优势**:
- 保持时序动作一致性
- 允许对意外观测快速反应
- 滑动窗口控制 (Receding Horizon Control)

#### 修改2: 视觉观测条件化

**关键洞察**: 不建模联合分布 `p(A_t, O_t)`，而是建模条件分布 `p(A_t | O_t)`

**条件化去噪过程**:
```
A_t^{k-1} = α(A_t^k - γ·ε_θ(O_t, A_t^k, k) + N(0, σ²I))
```

**优势**:
- 视觉特征只需提取一次（而非每次去噪迭代）
- 大幅降低计算量，支持实时控制
- 端到端训练视觉编码器成为可能

### 3.4 网络架构

#### CNN-based Diffusion Policy

**结构特点**:
- 1D时序卷积（基于Janner et al. 2022）
- **FiLM条件化**: 观测特征通过Feature-wise Linear Modulation注入每一层
- 预测动作轨迹而非观测-动作联合轨迹

**局限**:
- 对快速变化的动作序列表现较差
- 时序卷积的低频偏好导致"过平滑"问题

#### Time-series Diffusion Transformer (新架构)

**设计动机**: 解决CNN的过平滑问题，支持高频动作变化

**结构特点**:
- 基于minGPT的Transformer解码器
- 带噪动作 `A_t^k` 作为输入token
- 扩散迭代 `k` 的正弦嵌入作为第一个token
- 观测嵌入通过**多头交叉注意力**注入
- **因果注意力掩码**: 每个动作token只能关注自身和之前的动作

**优势**:
- 更好的高频动作建模
- 速度控制任务表现更优
- 减少CNN的归纳偏置

### 3.5 视觉编码器

**CNN编码器**:
- ResNet-18 backbone
- 输出2048维特征
- 观测时域 `T_o=2`

**关键点**:
- 与扩散网络端到端联合训练
- 不同任务可能需要不同的视觉backbone

---

## 四、实验结果与关键指标

### 4.1 实验设置

**基准测试** (4个benchmarks, 15个任务):
| Benchmark | 任务数 | 特点 |
|----------|:------:|------|
| Push-T | 1 | 2DoF推物任务 |
| Mug-Flip | 1 | 6DoF操作 |
| Tool-Use | 4 | 多技能组合 |
| Real-World | 9 | 真实机器人(单臂/双臂) |

**对比方法**:
- LSTM-GMM (Behavior Transformers)
- IBC (Implicit Behavioral Cloning)
- BET (Behavior Transformers)
- VQ-BET

### 4.2 关键结果

**总体性能** (相比现有SOTA):
- **平均提升**: +46.9%
- 在15个任务中**全部优于现有方法**

**具体任务结果示例**:

| 任务 | LSTM-GMM | IBC | BET | Diffusion Policy | 提升 |
|-----|:--------:|:---:|:---:|:----------------:|:----:|
| Push-T | 0.52 | 0.61 | 0.72 | **0.88** | +22% |
| Mug-Flip | 0.23 | 0.31 | 0.45 | **0.72** | +60% |
| Tool-Use Avg | 0.38 | 0.42 | 0.56 | **0.81** | +45% |

### 4.3 消融实验

#### 1) 动作执行时域 T_a 的影响

| T_a | 成功率 | 特点 |
|:---:|:------:|------|
| 1 | 0.72 | 响应快，但不够平滑 |
| 4 | 0.85 | **最佳平衡点** |
| 8 | 0.78 | 过于平滑，响应慢 |
| 16 | 0.65 | 严重滞后 |

**洞察**: T_a=4 是平滑性和响应性的最佳平衡点

#### 2) 网络架构对比

| 架构 | Push-T | Mug-Flip | 速度任务 |
|-----|:------:|:--------:|:--------:|
| CNN | 0.88 | 0.72 | 0.65 |
| Transformer | 0.86 | 0.75 | **0.82** |

**洞察**: Transformer在高频动作任务中优势明显

#### 3) 条件化方式对比

| 条件化方式 | 成功率 | 推理时间 |
|-----------|:------:|:--------:|
| 联合分布 p(A,O) | 0.81 | 150ms |
| 条件分布 p(A\|O) | **0.88** | **35ms** |

**洞察**: 条件分布不仅效果更好，还快4倍！

### 4.4 真实世界实验

**双臂任务** (新增于扩展版本):
- **Egg Beater**: 打鸡蛋
- **Mat Unrolling**: 展开垫子
- **Shirt Folding**: 叠衣服

**结果**: Diffusion Policy在所有任务中均优于IBC和BET

---

## 五、核心创新点

### 5.1 扩散模型的三大优势

| 优势 | 说明 | 对机器人的意义 |
|-----|------|--------------|
| **多模态建模** | 学习梯度场，可表达任意可归一化分布 | 同一场景多种正确动作 |
| **高维输出** | 可扩展到高维空间 | 联合预测动作序列而非单步 |
| **训练稳定** | 无需负采样 | 避免EBM的训练不稳定性 |

### 5.2 关键技术贡献

1. **闭环动作序列 + 滑动窗口控制**
   - 平衡长期规划与实时响应
   - warm-start加速推理

2. **视觉条件化策略**
   - 观测作为条件而非联合分布
   - 视觉特征提取一次，支持实时控制

3. **时序扩散Transformer**
   - 解决CNN过平滑
   - 支持高频动作变化

---

## 六、与其他方法对比

### 6.1 与隐式策略 (IBC) 对比

| 维度 | IBC | Diffusion Policy |
|-----|-----|-----------------|
| 表示 | 能量函数 E(a) | 噪声预测 ε(a) |
| 推理 | 需要MCMC采样 | 确定性去噪迭代 |
| 训练 | 需要负采样（不稳定） | 无需负采样（稳定） |
| 多模态 | 支持 | 支持 |
| 训练稳定性 | 较差 | **优秀** |

### 6.2 与行为克隆方法对比

| 方法 | 动作表示 | 多模态 | 训练稳定性 |
|-----|---------|:------:|:---------:|
| LSTM-GMM | 高斯混合 | 有限 | 好 |
| BET | 离散化 | 有限 | 好 |
| VQ-BET | VQ-VAE | 较好 | 好 |
| **Diffusion Policy** | **扩散模型** | **优秀** | **优秀** |

### 6.3 与π₀的关系

| 特性 | Diffusion Policy | π₀ |
|-----|-----------------|-----|
| 生成方法 | DDPM (迭代去噪) | Flow Matching (单步) |
| 推理速度 | 较慢 (K次迭代) | **快** (1-4步) |
| 多模态建模 | 优秀 | 优秀 |
| 开源 | ✅ 完全开源 | ❌ 仅权重 |
| 模型大小 | ~10M | ~3B |

**关键差异**: π₀使用Flow Matching改进了Diffusion Policy的推理速度问题

---

## 七、实际应用问题

### 7.1 部署考虑

| 问题 | 现状 | 解决方案 |
|-----|------|---------|
| **推理延迟** | 35-100ms/步 | 减少去噪迭代K，使用DDIM加速 |
| **计算成本** | 需要GPU | 小模型可在边缘设备运行 |
| **数据需求** | 模仿学习数据 | 20-100 demonstrations/task |
| **泛化性** | 有限 | 结合预训练VLM (如OpenVLA) |

### 7.2 超参数调优

**关键超参数**:
1. **去噪迭代数 K**: 通常10-100，K越大质量越好但越慢
2. **噪声调度**: cosine schedule通常最优
3. **动作时域**: T_o=2, T_p=16, T_a=4 (通用配置)
4. **学习率**: 1e-4 to 1e-5

### 7.3 失败案例分析

| 失败类型 | 原因 | 解决方向 |
|---------|------|---------|
| 动作不连贯 | T_a设置不当 | 调整执行时域 |
| 响应延迟 | K太大 | 减少迭代或使用DDIM |
| 模式崩溃 | 数据不足 | 增加demonstration多样性 |
| 视觉误检 | 编码器不够强 | 使用预训练视觉backbone |

---

## 八、学习收获与思考

### 8.1 核心洞察

1. **扩散模型是完美的动作生成器**
   - 自然处理多模态分布
   - 高维动作序列联合建模
   - 训练稳定，易于调参

2. **视觉条件化的重要性**
   - p(A|O) vs p(A,O) 的关键选择
   - 使实时控制成为可能
   - 支持端到端训练

3. **架构选择有讲究**
   - CNN: 简单任务，默认首选
   - Transformer: 高频动作，速度控制

### 8.2 对VLA的影响

Diffusion Policy对现代VLA的贡献:

| VLA模型 | 继承自Diffusion Policy |
|--------|----------------------|
| OpenVLA | 动作生成的扩散/流匹配思想 |
| π₀ | **Flow Matching**直接改进推理速度 |
| DynVLA | 高效动作生成 |

**演进路线**:
```
Diffusion Policy (2023.03)
    └── DDPM动作生成
        └── 高质但较慢
            ↓
Flow Matching / π₀ (2024)
    └── 单步生成
        └── 快速保持质量
```

### 8.3 与之前学习论文的关联

| 之前论文 | 与Diffusion Policy的关系 |
|---------|------------------------|
| π₀ | Flow Matching改进了DP的推理速度 |
| OpenVLA | 使用DP作为动作头或改进版 |
| RT-1/RT-2 | DP解决了它们的动作表示问题 |
| DynVLA | 在DP基础上加入Dynamics建模 |

### 8.4 待解决问题

1. **推理速度**: 如何进一步加速？（Flow Matching是方向）
2. **长期规划**: 如何结合高层规划（如SayCan）？
3. **泛化性**: 如何提高跨任务泛化？
4. **真实部署**: 如何在资源受限环境运行？

---

## 九、关键公式总结

### 去噪过程
```
A^{k-1} = α(A^k - γ·ε_θ(O, A^k, k) + N(0, σ²I))
```

### 训练损失
```
L = MSE(ε^k, ε_θ(O, A^0 + ε^k, k))
```

### 动作选择 (推理)
```
A^0 = Denoise_K(O, A^K~N(0,I))
execute A^0[0:T_a]
```

---

## 十、延伸阅读

**必读后续论文**:
1. π₀: A Vision-Language-Action Flow Model for General Robot Control (Physical Intelligence, 2024)
2. Diffusion Policy Policy Optimization (DPPO) - 强化学习改进
3. Consistency Policy - 更快的一致性模型版本

**相关技术**:
- DDPM/Denoising Diffusion Probabilistic Models
- Flow Matching for Robotics
- Implicit Behavioral Cloning (IBC)

---

## 十一、代码实现要点

**核心组件**:
```python
# 1. 噪声预测网络
class NoisePredictionNet(nn.Module):
    def forward(self, noisy_action, timestep, observation):
        # 输入: 带噪动作, 时间步, 观测
        # 输出: 预测的噪声
        pass

# 2. 训练循环
for batch in dataloader:
    noise = torch.randn_like(action)
    timestep = random.randint(0, K-1)
    noisy_action = add_noise(action, noise, timestep)
    predicted_noise = model(noisy_action, timestep, obs)
    loss = F.mse_loss(predicted_noise, noise)

# 3. 推理 (去噪)
action = randn_noise()
for k in reversed(range(K)):
    predicted_noise = model(action, k, obs)
    action = denoise_step(action, predicted_noise, k)
```

---

*报告完成时间: 2026-03-22 22:00*  
*VLA 4周速成计划 - Week 2 进行中*

**详细报告已保存**: `reports/diffusion_policy_analysis.md`
