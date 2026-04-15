# VLA & 世界模型 4周速成计划 - 完整总结

**计划周期**: 2026-03-13 至 2026-04-13 (4周)  
**总报告数**: 39篇  
**生成日期**: 2026-04-13

---

## 📚 总索引/目录

### VLA 核心模型 (16篇)
| 序号 | 论文 | 会议/年份 | 关键词 |
|:---:|:---|:---:|:---|
| 1 | OpenVLA | arXiv 2024 | 7B开源, DINOv2+SigLIP |
| 2 | DynVLA | arXiv 2024 | Dynamics CoT, 动态推理 |
| 3 | Helix (Figure AI) | 2024 | System 1+2, 人形机器人 |
| 4 | π₀ (Pi-Zero) | arXiv 2024 | Flow Matching, 3B参数 |
| 5 | SmolVLA | arXiv 2024 | 小模型, 10倍压缩 |
| 6 | StructVLA | arXiv 2024 | 结构化帧预测 |
| 7 | ReMem-VLA | arXiv 2024 | 双层循环记忆 |
| 8 | SayCan | Google 2022 | Affordance Grounding |
| 9 | Inner Monologue | arXiv 2022 | 闭环语言规划 |
| 10 | π₀-FAST | arXiv 2025 | DCT动作分词 |
| 11 | DySL-VLA | arXiv 2026 | 动态-静态层跳过 |
| 12 | ACoT-VLA | arXiv 2024 | 动作空间思维链 |
| 13 | VLA-MBPO | arXiv 2024 | 世界模型+RL微调 |
| 14 | VLA机制研究 | ICLR Workshop 2025 | 视觉主导发现 |
| 15 | π*₀.6 (VLA+RL) | arXiv 2025 | RECAP框架 |
| 16 | RT-2 | CoRL 2022 | 55B参数, Web知识 |

### 世界模型与MPC (11篇)
| 序号 | 论文 | 会议/年份 | 关键词 |
|:---:|:---|:---:|:---|
| 17 | RAE-NWM | arXiv 2024 | DINOv2密集表示 |
| 18 | RWM-U + MOPO-PPO | arXiv 2024 | 不确定性感知 |
| 19 | WMPC | arXiv 2024 | 多模态预测控制 |
| 20 | Robust Convex MPC | arXiv 2024 | Tube MPC, 走廊规划 |
| 21 | R2-Dreamer | ICLR 2026 | 冗余减少, 无解码器 |
| 22 | DreamerV4 | arXiv 2025 | Shortcut Forcing, 实时 |
| 23 | RSSM详解 | 技术报告 | Dreamer核心架构 |
| 24 | GWM | arXiv 2024 | 3D Gaussian世界模型 |
| 25 | NVIDIA Cosmos | 2024 | 物理AI平台 |
| 26 | V-JEPA 2 | arXiv 2025 | Meta自监督视频模型 |
| 27 | Genie 3 | DeepMind 2025 | 交互式世界模型 |

### 视频生成与规划 (4篇)
| 序号 | 论文 | 会议/年份 | 关键词 |
|:---:|:---|:---:|:---|
| 28 | UniPi | NeurIPS 2023 | Policy-as-Video |
| 29 | RoboDreamer | ICML 2024 | 组合式世界模型 |
| 30 | LingBot-VA | arXiv 2024 | 因果世界模型 |
| 31 | PhysWorld | arXiv 2024 | 物理视频生成 |

### 灵巧手操作 (5篇)
| 序号 | 论文 | 会议/年份 | 关键词 |
|:---:|:---|:---:|:---|
| 32 | OpenAI In-Hand | RSS 2018 | RL+Sim-to-Real |
| 33 | DexPilot | RSS 2020 | 视觉遥操作 |
| 34 | DIGIT | ICRA 2020 | 低成本触觉传感器 |
| 35 | RT-2 (灵巧手) | CoRL 2022 | VLA大模型 |
| 36 | Qin 2022 | arXiv 2022 | 单相机遥操作 |

### 产业调研报告 (3篇)
| 序号 | 报告 | 主题 |
|:---:|:---|:---|
| 37 | 全球人形机器人公司调研 | 18家公司深度分析 |
| 38 | Agility Robotics深度报告 | 首家商业化公司 |
| 39 | 机器人数据采集与测试 | 数据策略与验证方法 |

---

## 🎯 按主题分类核心洞察

### 主题1: VLA模型架构演进

**三代架构演进**:
```
Gen 1 (2022): RT-1 → RT-2
  - 自回归动作生成
  - 端到端训练
  
Gen 2 (2024): OpenVLA, π₀, Helix
  - 双流视觉编码器 (DINOv2 + SigLIP)
  - 多样化动作生成机制 (自回归/Flow Matching/Diffusion)
  
Gen 3 (2025+): π₀-FAST, DySL-VLA
  - 推理加速优化
  - 稀疏/动态推理
```

**关键洞察**:
1. **视觉编码器**: DINOv2 特征 > CLIP/ViT，密集特征 > 全局特征
2. **动作生成机制**: Flow Matching (π₀) 在效率和质量间取得最佳平衡
3. **规模法则**: 7B参数是实用性与性能的最佳平衡点

### 主题2: 世界模型技术路线

**四大技术路线对比**:

| 路线 | 代表模型 | 核心思想 | 优势 | 劣势 |
|:---|:---|:---|:---|:---|
| **隐式潜在模型** | DreamerV3/V4, RSSM | 学习压缩的潜在状态 | 计算高效 | 可解释性差 |
| **显式3D表示** | GWM | 3D Gaussian表示 | 空间理解强 | 计算开销大 |
| **视频生成模型** | UniPi, RoboDreamer, Cosmos | 像素空间预测 | 直观、通用 | 推理慢 |
| **自监督学习** | V-JEPA 2 | 特征空间预测 | 数据效率高 | 需要大量预训练 |

**关键洞察**:
1. **表示空间**: DINOv2 密集特征逐渐成为共识
2. **预测目标**: 结构化帧预测 > 密集视频预测 > 抽象语义
3. **不确定性建模**: 长程预测必须考虑不确定性累积

### 主题3: 组合式与层次化设计

**组合性在不同层面的应用**:

| 层级 | 应用 | 代表工作 |
|:---|:---|:---|
| **语言组合** | 指令解析为原语 | RoboDreamer |
| **动作组合** | 动作序列分解 | π₀-FAST (DCT分词) |
| **视觉组合** | 场景分解为对象 | GWM |
| **模态组合** | 文本+图像+草图 | RoboDreamer |

**关键洞察**:
- 组合性是实现零样本泛化的关键
- 自然语言的组合性可以迁移到视觉和动作空间

### 主题4: 推理加速技术

**加速技术对比**:

| 技术 | 方法 | 加速比 | 代表 |
|:---|:---|:---:|:---|
| **Token跳过** | 动态-静态层跳过 | 3.75× | DySL-VLA |
| **动作分词** | DCT压缩+自回归 | 5-10× | π₀-FAST |
| **Shortcut Forcing** | 跳过未来表示 | 10×+ | DreamerV4 |
| **稀疏推理** | 仅推理必要token | 2-3× | DynVLA |

### 主题5: Sim-to-Real与真实部署

**关键挑战与解决方案**:

| 挑战 | 解决方案 | 代表工作 |
|:---|:---|:---|
| 视觉域差距 | DINOv2预训练特征 | OpenVLA, RAE-NWM |
| 动作分布偏移 | 离线RL + 世界模型 | VLA-MBPO, RWM-U |
| 数据稀缺 | 视频生成数据增强 | UniPi, RoboDreamer |
| 安全约束 | Tube MPC | Robust Convex MPC |

---

## 📊 关键技术对比表格

### 表1: VLA模型对比

| 模型 | 参数量 | 视觉编码 | 动作生成 | 推理速度 | 开源 |
|:---:|:---:|:---|:---|:---:|:---:|
| RT-2 | 55B | ViT | 自回归 | 慢 | ❌ |
| OpenVLA | 7B | DINOv2+SigLIP | 自回归 | 中等 | ✅ |
| π₀ | 3B | ViT | Flow Matching | 快 | ❌ |
| SmolVLA | 0.5B | DINOv2+SigLIP | 自回归 | 快 | ✅ |
| Helix | 7B+80M | 双流 | 自回归 | 中等 | ❌ |
| π₀-FAST | 3B | ViT | DCT+自回归 | 很快 | ❌ |

### 表2: 世界模型对比

| 模型 | 架构 | 表示 | 预测目标 | 交互性 | 实时性 |
|:---:|:---|:---|:---|:---:|:---:|
| DreamerV3 | RSSM | 隐式潜在 | 潜在状态+奖励 | ✅ | ⚠️ |
| DreamerV4 | RSSM+Shortcut | 隐式潜在 | 潜在状态 | ✅ | ✅ |
| GWM | Transformer | 3D Gaussian | 下一帧 | ✅ | ❌ |
| Cosmos | Transformer | 隐式神经 | 视频 | ✅ | ⚠️ |
| V-JEPA 2 | JEPA | 隐式潜在 | 特征预测 | ✅ | ✅ |
| UniPi | Video Diffusion | 像素 | 视频序列 | ✅ | ❌ |
| RoboDreamer | Video Diffusion | 像素 | 视频序列 | ✅ | ❌ |

### 表3: 动作生成机制对比

| 机制 | 原理 | 优点 | 缺点 | 适用场景 |
|:---|:---|:---|:---|:---|
| **自回归** | 逐token预测 | 与LLM统一 | 慢、误差累积 | 通用 |
| **Flow Matching** | 连续时间流 | 10步高效 | 训练复杂 | 高频控制 |
| **Diffusion** | 去噪生成 | 多模态好 | 慢 | 需要多样性 |
| **DCT分词** | 频域压缩 | 紧凑表示 | 信息损失 | 平滑轨迹 |
| **Dynamics CoT** | 推理时展开 | 可解释 | 开销大 | 复杂推理 |

---

## 📝 学习总结与关键结论

### 1. 核心技术趋势

**VLA领域**:
- 从单一自回归向多样化生成机制演进
- 7B参数成为开源社区的黄金标准
- 视觉编码器从通用CLIP向任务专用DINOv2迁移

**世界模型领域**:
- 从隐式潜在向显式/混合表示发展
- 视频生成模型成为新范式，但推理速度仍是瓶颈
- 物理真实性和组合泛化是下一代关键挑战

### 2. 关键技术洞察

**Insight 1: 表示学习的重要性**
```
DINOv2 密集特征 > VAE/CLIP > 原始像素
```
统一的视觉表示是VLA和世界模型的基础。

**Insight 2: 组合性 = 泛化能力**
- 语言的组合性 → 视频生成组合性 (RoboDreamer)
- 动作的组合性 → 紧凑表示 (π₀-FAST)
- 场景的组合性 → 结构化理解 (GWM)

**Insight 3: 层次化规划的必要性**
- 粗到细的时间超分辨率 (UniPi)
- 高频动作vs低频推理的分离 (DySL-VLA)
- 规划与执行的解耦

**Insight 4: 推理速度是部署瓶颈**
- 实时性是真实机器人部署的关键
- 稀疏推理和Token跳过成为标准优化手段

### 3. 实践建议

**对于研究者**:
1. 优先掌握 DINOv2 + LoRA 微调流程
2. 关注视频扩散模型在机器人中的应用
3. 组合泛化和物理真实性是未来热点

**对于工程师**:
1. 7B参数VLA是当前最佳性价比选择
2. 分层架构（规划+控制）比端到端更实用
3. 数据效率比模型规模更重要

### 4. 未来方向

**短期 (6-12个月)**:
- 视频世界模型的推理加速
- 多模态VLA（视觉+语言+触觉）
- 小模型高效微调方法

**中期 (1-2年)**:
- 物理真实的视频生成
- 跨本体（cross-embodiment）泛化
- 在线学习与自适应

**长期 (3-5年)**:
- 通用世界模型平台
- 自主数据收集与闭环学习
- 物理常识推理

---

## 📂 报告文件清单

```
workspace/reports/
├── VLA核心模型 (16篇)
│   ├── openvla-arxiv-2406.09246.md
│   ├── dynvla-arxiv-report.md
│   ├── helix-figure-ai-report.md
│   ├── pi0-pi-zero-report.md
│   ├── smolvla-arxiv-report.md
│   ├── structvla-arxiv-report.md
│   ├── remem-vla-arxiv-report.md
│   ├── saycan-google-report.md
│   ├── inner-monologue-report.md
│   ├── pi0-fast-arxiv-2501.09747.md
│   ├── dysl-vla-arxiv-2602.22896.md
│   ├── acot-vla-arxiv-report.md
│   ├── vla-mbpo-arxiv-report.md
│   ├── vla-mechanism-iclr-workshop-report.md
│   ├── pi-star-06-vla-rl-report.md
│   └── rt2-corl-2022-report.md
├── 世界模型与MPC (11篇)
│   ├── rae-nwm-arxiv-report.md
│   ├── rwm-u-mopo-ppo-report.md
│   ├── wmpc-multimodal-adaptation-report.md
│   ├── robust-convex-mpc-manipulators-report.md
│   ├── r2-dreamer-iclr-2026-report.md
│   ├── dreamerv4-world-model-report.md
│   ├── rssm-recurrent-state-space-model-report.md
│   ├── gwm-3d-gaussian-world-model-report.md
│   ├── nvidia-cosmos-physic-ai-report.md
│   ├── v-jepa2-meta-report.md
│   └── genie3-deepmind-report.md
├── 视频生成与规划 (4篇)
│   ├── unipi-neurips-2023-report.md
│   ├── robodreamer-icml-2024-report.md
│   ├── lingbot-va-causal-world-model-report.md
│   └── physworld-video-generation-report.md
├── 灵巧手操作 (5篇)
│   ├── openai-in-hand-manipulation-report.md
│   ├── dexpilot-teleoperation-report.md
│   ├── digit-tactile-sensor-report.md
│   ├── rt2-dexterous-hand-report.md
│   └── qin2022-single-camera-report.md
└── 产业调研 (3篇)
    ├── humanoid-robot-companies-research-report.md
    ├── agility-robotics-deep-dive-report.md
    ├── robot-data-collection-methods-report.md
    └── robot-testing-validation-report.md
```

---

## 🎓 学习建议

### 入门路径
1. **Week 1 基础**: OpenVLA + RSSM + RT-2
2. **Week 2 深入**: π₀ + DreamerV4 + SayCan
3. **Week 3 世界模型**: UniPi + GWM + Cosmos
4. **Week 4 前沿**: RoboDreamer + V-JEPA 2 + Genie 3

### 深入阅读优先级
1. ⭐⭐⭐⭐⭐: OpenVLA, π₀, DreamerV4, UniPi, RoboDreamer
2. ⭐⭐⭐⭐: DySL-VLA, π₀-FAST, R2-Dreamer, GWM, V-JEPA 2
3. ⭐⭐⭐: 其他模型和调研报告

---

**总结生成时间**: 2026-04-13  
**总阅读量**: 39篇论文/报告  
**建议复习周期**: 每2周回顾一次核心洞察

*祝学习愉快！*
