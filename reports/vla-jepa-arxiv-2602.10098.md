# VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model

**论文分析报告**

---

## 📋 基本信息

| 属性 | 详情 |
|-----|------|
| **论文标题** | VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model |
| **arXiv编号** | arXiv:2602.10098v2 |
| **发布日期** | 2026年2月10日（v1），2月14日（v2） |
| **作者团队** | 中国科学技术大学、中关村实验室、上海交通大学、清华大学、东方理工大学、中国科学院大学、南开大学 |
| **通讯作者** | Zhibo Chen |
| **代码开源** | https://github.com/ginwind/VLA-JEPA/ |
| **项目主页** | https://ginwind.github.io/VLA-JEPA/ |
| **模型权重** | https://huggingface.co/ginwind/VLA-JEPA/ |

---

## 🎯 核心问题与动机

### 研究背景

当前VLA（Vision-Language-Action）模型的预训练面临一个关键挑战：**如何有效利用互联网规模的视频数据**来学习动作表示？

现有方法主要采用"latent action"（潜在动作）预训练范式，但作者指出这些方法存在根本性缺陷：

### 四大失败模式

| 问题 | 描述 | 后果 |
|-----|------|------|
| **1. 像素级目标偏差** | 目标函数关注视觉变化（纹理、光照、背景）而非动作相关状态转移 | 学习到高方差、低可控性的表示 |
| **2. 真实视频噪声放大** | 相机运动和非因果背景变化强于交互引起的状态变化 | 潜在动作变成delta-frame编码器 |
| **3. 信息泄漏** | 未来帧信息作为输入进入模型，导致捷径学习 | 潜在动作语义空洞，仅用于匹配训练损失 |
| **4. 多阶段pipeline复杂** | 需要表示预训练→latent action学习→策略学习三阶段 | 工程复杂、阶段间不一致 |

### 核心洞察

> 对于具身智能体，最有用的"动作"概念不是像素差异的紧凑描述，而是**捕获交互下底层状态如何演化的变量**（action-relevant state transition semantics）

---

## 🏗️ 方法详解

### 核心思想：Leakage-Free State Prediction

VLA-JEPA采用**JEPA（Joint-Embedding Predictive Architecture）**框架，核心设计原则：

```
目标编码器（Target Encoder）: 从未来帧生成潜在表示（监督信号）
                ↓
学生路径（Student Pathway）: 仅接收当前观测 + VLM Backbone
                ↓
预测器（Predictor）: 映射历史潜在状态 + latent action → 未来潜在状态
```

**关键约束**：未来帧仅用于构建训练目标，**绝不作为VLM Backbone的输入** —— 这消除了latent-action collapse的捷径。

### 架构组成

#### 1. 基础模型
- **VLM Backbone**: Qwen3-VL（基于Qwen3 + SigLIP-2视觉编码器）
- **世界状态编码器**: V-JEPA2自监督视频编码器

#### 2. 可学习Token设计
- `⟨latent_i⟩`: 第i个时间步的潜在动作token
- `⟨action⟩`: 动作token

#### 3. 世界状态编码（World State Encoder）

对于多视角视频，统一世界状态表示：

```
s_{t_i} = ||_v F(I_{v,t_i})

其中：
- F(·): 单视角视频编码器（V-JEPA2）
- ||: 向量拼接操作
- s_{t_i}: 时间戳t_i的统一世界状态表示
```

#### 4. Latent Action预训练（World Modeling）

**输入**: 多视角初始观测 + 语言指令

**处理流程**:
```
1. VLM将⟨latent_i⟩映射为潜在表示z_{t_i}
   z_{t_i} = p_θ^VLM(⟨latent_i⟩ | {I_{j,t_0}}_{j=0}^v, ℓ)

2. 世界模型预测下一状态块
   ŝ_{t_{1:i+1}} = p_θ^WM(s_{t_{0:i}}, z_{t_{0:i}})
```

#### 5. 训练目标

从JEPA视角，目标可解释为最大化预测对数似然的ELBO：

```
log p(s_{t_{1:T}} | z_{t_{0:T-1}}) ≥ 
  Σ_{k=1}^T E_{s_{t_k}~F(·)}[log p_θ(ŝ_{t_k} | s_{t_k})] 
  - D_KL[F(·) || p_θ^WM]
```

由于F(·)产生确定性嵌入，KL项消失，ELBO退化为重建目标。

#### 6. Flow-Matching动作头

对于机器人演示数据，集成基于flow-matching的动作生成器支持精确的末端执行器轨迹生成。

### 训练流程

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: JEPA预训练（人类视频 + 可选机器人数据）         │
│  - 人类视频：Alignment Loss                             │
│  - 机器人数据：Alignment Loss + Action Prediction Loss  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Action-Head微调                               │
│  - 端到端融合两个目标                                    │
│  - 利用学习到的状态转移动力学                            │
└─────────────────────────────────────────────────────────┘
```

---

## 🔬 实验设置

### 评估基准

| 基准 | 描述 |
|-----|------|
| **LIBERO** | 机器人操作基准测试套件 |
| **LIBERO-Plus** | LIBERO扩展版本 |
| **SimplerEnv** | 简化环境评估 |
| **Real-World** | 真实世界Franka机器人操作任务 |

### 对比方法

- RT系列（RT-1, RT-2, RT-H）
- OpenVLA
- π0, π0.5
- LAPA
- UniVLA
- MotoGPT
- 其他VLA方法

---

## 📊 实验结果

### 主要发现

VLA-JEPA在以下方面取得**一致的增益**：

1. **泛化能力（Generalization）**: 跨环境、跨物体、跨场景迁移
2. **鲁棒性（Robustness）**: 对相机运动、背景变化、光照变化的稳定性
3. **数据效率**: 利用无标注人类视频减少对action-labeled数据的依赖

### 关键优势

| 优势 | 说明 |
|-----|------|
| **语义鲁棒性** | 监督在潜在空间而非像素空间，对相机运动和背景变化鲁棒 |
| **简化流程** | 两阶段训练（JEPA预训练 → Action-head微调），无需多阶段pipeline |
| **无信息泄漏** | 未来帧仅作监督目标，消除了捷径学习 |
| **跨域训练** | 同时支持人类视频（无动作标签）和机器人数据（有动作标签） |

---

## 💡 核心创新点

### 1. 理论贡献

- **系统性分析了latent-action预训练的缺陷**: 指出了像素级目标、信息泄漏等根本问题
- **提出了action-relevant state transition semantics的原则**: 明确了对控制有用的"动作"应该捕获什么

### 2. 方法创新

- **首个将JEPA框架系统应用于VLA的架构**: 利用V-JEPA2作为世界状态编码器
- **Leakage-free设计**: 通过目标编码器-预测器分离消除信息泄漏
- **统一框架**: 支持action-free和action-labeled数据的联合训练

### 3. 实践贡献

- **简化的两阶段训练流程**: 相比之前的三阶段+方法更易实现
- **开源代码和模型**: 促进社区复现和后续研究

---

## 🔗 相关工作对比

### 与Latent Action方法的对比

| 方法 | 核心思想 | 局限性 | VLA-JEPA改进 |
|-----|---------|--------|-------------|
| **LAPA** | 从帧间差异提取latent action | 像素级shortcut | 潜在空间预测，无像素重建 |
| **UniVLA** | 统一codebook对齐人类和机器人视频 | 信息泄漏 | Leakage-free设计 |
| **MotoGPT** | 离散motion token预训练 | 多阶段pipeline | 两阶段简化流程 |
| **VILLA-X** | 人类+机器人视频联合训练 | 依赖delta frame | 基于世界模型的状态转移 |

### 与VLA方法的对比

| 方法 | 依赖数据 | 预训练策略 |
|-----|---------|-----------|
| **RT-2** | 大规模action-labeled数据 | 端到端微调 |
| **OpenVLA** | 大规模action-labeled数据 | 视觉-语言预训练 + 动作微调 |
| **π0** | 大规模action-labeled数据 | Flow matching动作生成 |
| **VLA-JEPA** | 人类视频（无标签）+ 少量机器人数据 | JEPA世界模型预训练 |

---

## 🎯 关键洞见与启示

### 1. 从像素到语义的转变

VLA-JEPA代表了机器人学习范式的转变：

```
传统方法: Pixel Prediction → 易受外观、光照、背景干扰
    ↓
VLA-JEPA: Latent Representation Prediction → 关注语义/动力学
```

### 2. 世界模型在VLA中的价值

- **不仅仅是表示学习**: 世界模型提供了对状态转移动力学的显式建模
- **跨域迁移**: 从人类视频学到的动力学知识可迁移到机器人控制
- **数据效率**: 减少对昂贵机器人数据的依赖

### 3. 信息泄漏的危害

论文深刻揭示了信息泄漏如何破坏latent action学习：
- 当未来帧作为输入时，模型会走捷径
- 学到的"action"只是未来帧的压缩表示
- 这种表示对控制毫无意义

### 4. JEPA架构的普适性

从V-JEPA（视频理解）→ I-JEPA（图像理解）→ VL-JEPA（视觉-语言）→ **VLA-JEPA（视觉-语言-动作）**

JEPA架构正逐步扩展到具身智能领域，展现强大的扩展性。

---

## ⚠️ 局限性与未来方向

### 当前局限

1. **依赖V-JEPA2**: 世界状态编码器的性能直接影响VLA-JEPA
2. **计算成本**: JEPA预训练需要较大计算资源
3. **长程规划**: 论文主要关注短程操作任务，长程任务表现待验证

### 未来方向

1. **扩展到导航和移动操作**: 测试在更复杂场景下的表现
2. **结合在线学习**: 在部署后持续改进世界模型
3. **多模态世界模型**: 整合触觉、力觉等更多感知模态
4. **与扩散模型结合**: 探索JEPA与diffusion policy的协同

---

## 📚 相关论文

### 基础论文
- **V-JEPA**: V-JEPA: Video Joint Embedding Predictive Architecture (Meta, 2024)
- **V-JEPA 2**: V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (Meta, 2025)
- **I-JEPA**: I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (Meta, 2023)

### VLA相关
- **RT-2**: RT-2: Vision-Language-Action Models (Google DeepMind, 2023)
- **OpenVLA**: OpenVLA: An Open-Source Vision-Language-Action Model (2024)
- **π0**: π0: A Vision-Language-Action Flow Model for General Robot Control (Physical Intelligence, 2024)

### Latent Action相关
- **LAPA**: Latent Action Pretraining from Videos (2024)
- **UniVLA**: Learning Universal Policies via Text-Guided Video Generation (2025)
- **MotoGPT**: Language-Model-Based Robot Planning and Action Generation (2024)

---

## 🏁 总结

VLA-JEPA代表了VLA模型预训练的重要进展。通过引入JEPA框架，论文解决了现有latent-action方法的四大核心问题：

1. ✅ **消除像素级偏差** → 潜在空间预测
2. ✅ **抑制噪声运动** → 语义级世界状态编码
3. ✅ **防止信息泄漏** → Leakage-free设计
4. ✅ **简化训练流程** → 两阶段pipeline

**核心贡献**在于提出了一种新的VLA预训练范式：不是从像素变化中学习"动作"，而是在潜在空间中预测"状态转移"。这一转变使模型更关注控制相关的语义动力学，而非表面的视觉变化。

对于机器人学习领域，VLA-JEPA提供了一条减少对昂贵机器人数据依赖、更好利用互联网视频数据的可行路径。

---

## 📖 引用信息

```bibtex
@article{sun2026vlajepa,
  title={VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model},
  author={Sun, Jingwen and Zhang, Wenyao and Qi, Zekun and Ren, Shaojie and Liu, Zezhi and Zhu, Hanxin and Sun, Guangzhong and Jin, Xin and Chen, Zhibo},
  journal={arXiv preprint arXiv:2602.10098},
  year={2026}
}
```

---

*报告生成日期: 2026年4月13日*
*分析师: 小小*
