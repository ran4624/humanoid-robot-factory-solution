# MEMORY.md - 长期记忆档案

_核心记忆与重要信息的精选整理_

---

## 👤 用户档案

| 属性 | 信息 |
|-----|------|
| **姓名** | 旭哥 |
| **称呼** | 旭哥 |
| **时区** | Asia/Shanghai (北京时间) |
| **身份** | VLA & 机器人技术学习者/研究者 |

---

## 📚 学习项目：VLA & 世界模型 4周速成计划

### 项目概览
- **开始日期**: 2026-03-13
- **计划周期**: 4周
- **每日投入**: 晚上10点，2-3小时
- **当前状态**: ✅ Week 1 & Week 2 已完成，准备进入 Week 3 (代码复现+项目实战)

### 学习路线图

#### Week 1: VLA基础 ✅ 已完成
| 日期 | 论文/主题 | 状态 |
|:---:|----------|:---:|
| 3/13 | RT-2 (Google DeepMind) | ✅ 深度分析 |
| 3/14 | RT-2 复习 | ✅ 复习巩固 |
| 3/15 | OpenVLA + PhysWorld | ✅ 深度分析 |
| 3/17 | π₀ (Pi-Zero) + SmolVLA + RoboDreamer | ✅ 深度分析 |
| 3/19 | OpenVLA + RT-2 + DreamerV3 | ✅ 复习+补全 |

#### Week 2: VLA深入 ✅ 已完成
| 日期 | 论文/主题 | 状态 |
|:---:|----------|:---:|
| 3/20 | π₀-FAST (动作分词) | ✅ 深度分析 |
| 3/20 | DySL-VLA (推理加速) | ✅ 深度分析 |
| 3/20 | WMPC (世界模型预测控制) | ✅ 深度分析 |
| 3/21 | Inner Monologue / SayCan | ✅ 深度分析 |
| 3/22 | 人形机器人公司调研 | ✅ 完成18家公司 |
| 3/22 | 灵巧手操作算法综述 | ✅ 5篇核心论文 |
| 3/24 | R2-Dreamer (冗余减少世界模型) | ✅ 深度分析 |
| 3/24 | π*₀.6 (VLA+RL) | ✅ 深度分析 |
| 3/25 | DreamerV4 (实时世界模型) | ✅ 深度分析 |
| 3/25 | 机器人数据采集方法 | ✅ 调研报告 |
| 3/25 | 机器人功能验证测试方法 | ✅ 调研报告 |
| 3/26 | RSSM架构详解 | ✅ 技术报告 |

#### Week 3: 世界模型深入 + 代码复现 🔄 已完成
- Day 14-15 (3/26-27): VLA机制研究 + LingBot-VA因果世界模型 ✅
- Day 16-18 (4/1-3): Cosmos + GWM世界模型 ✅
- Day 19-20 (4/6): ACoT-VLA + VLA-MBPO ✅

#### Week 4: 前沿模型学习 ✅ 已完成
- Day 21-22 (4/3): GWM + Cosmos深度分析 ✅
- Day 23-24 (4/6): ACoT-VLA + VLA-MBPO 世界模型RL ✅
- Day 25 (4/8): V-JEPA 2 + Genie 3 最新世界模型 ✅
- Day 26 (4/13): UniPi + RoboCrafter-QA 补充收官 ✅

---

## 📝 已生成详细报告的论文/报告（39篇）

### VLA核心模型（11篇）

| 序号 | 论文 | 核心贡献 | 日期 |
|:---:|------|---------|:---:|
| 1 | **OpenVLA** | 7B开源VLA，DINOv2+SigLIP双编码器 | 3/19 |
| 2 | **DynVLA** | Dynamics CoT，动态token推理 | 3/19 |
| 3 | **Helix** | Figure AI人形，System 1+2双系统 | 3/19 |
| 4 | **π₀ (Pi-Zero)** | Flow Matching动作生成，3B参数 | 3/17 |
| 5 | **SmolVLA** | 小模型VLA，10倍参数压缩 | 3/18 |
| 6 | **StructVLA** | 结构化帧预测规划 | 3/17 |
| 7 | **ReMem-VLA** | 双层循环记忆增强 | 3/17 |
| 8 | **SayCan** | Affordance Grounding，LLM+Value Function | 3/21 |
| 9 | **Inner Monologue** | 闭环语言规划，感知反馈 | 3/21 |
| 10 | **π₀-FAST** | DCT动作分词，自回归VLA加速 | 3/20 |
| 11 | **DySL-VLA** | 动态-静态层跳过，3.75×推理加速 | 3/20 |

### 世界模型与MPC（8篇）

| 序号 | 论文 | 核心贡献 | 日期 |
|:---:|------|---------|:---:|
| 12 | **RAE-NWM** | DINOv2密集表示导航世界模型 | 3/18 |
| 13 | **RWM-U + MOPO-PPO** | 不确定性感知离线MBRL | 3/18 |
| 14 | **WMPC** | 世界模型预测控制+多模态适应 | 3/20 |
| 15 | **Robust Convex MPC** | 鲁棒Tube MPC+走廊规划 | 3/20 |
| 23 | **R2-Dreamer** | 冗余减少世界模型，无需解码器/DA | 3/24 |
| 24 | **DreamerV4** | Shortcut Forcing，单GPU实时世界模型 | 3/25 |
| 25 | **RSSM详解** | Dreamer核心架构深度解析 | 3/26 |
| 26 | **π*₀.6** | RECAP框架，VLA从经验中学习 | 3/24 |
| 29 | **VLA机制研究** | 大规模机制分析，视觉主导发现 | 3/26 |
| 30 | **LingBot-VA** | 因果世界模型，自回归扩散+闭环 | 3/26 |
| 31 | **GWM** | 3D Gaussian世界模型，显式表示 | 4/3 |
| 32 | **NVIDIA Cosmos** | 物理AI世界基础模型平台 | 4/3 |
| 33 | **ACoT-VLA** | 动作空间思维链推理 | 4/6 |
| 34 | **VLA-MBPO** | 世界模型+VLA的RL微调 | 4/6 |
| 35 | **V-JEPA 2** | Meta自监督视频世界模型 | 4/8 |
| 36 | **Genie 3** | DeepMind交互式世界模型 | 4/8 |
| 37 | **UniPi** | Policy-as-Video范式，文本引导视频生成 | 4/13 |
| 38 | **RoboCrafter-QA** | LLM软体机器人设计评估基准 | 4/13 |
| 39 | **RoboDreamer** | 组合式世界模型，语言组合性→视频组合性 | 4/13 |

### 灵巧手操作（5篇核心论文）

| 序号 | 论文 | 核心贡献 | 日期 |
|:---:|------|---------|:---:|
| 16 | **OpenAI In-Hand** | RL+Sim-to-Real，Shadow Hand操作 | 3/22 |
| 17 | **DexPilot** | 视觉遥操作，低成本数据收集 | 3/22 |
| 18 | **DIGIT** | 低成本触觉传感器，开源 | 3/22 |
| 19 | **RT-2** | VLA大模型，互联网知识迁移 | 3/22 |
| 20 | **Qin 2022** | 单相机遥操作+模仿学习 | 3/22 |

### 产业调研报告（4篇）

| 序号 | 报告 | 核心内容 | 日期 |
|:---:|------|---------|:---:|
| 21 | **全球人形机器人公司调研** | 18家公司，融资/技术/商业化 | 3/22 |
| 22 | **Agility Robotics深度报告** | 首家商业化公司全解析 | 3/22 |
| 27 | **机器人数据采集方法调研** | 遥操作/自主采集/视频学习技术路线 | 3/25 |
| 28 | **机器人功能验证测试方法** | 国内外公司测试规模与标准体系 | 3/25 |

---

## 🎯 核心技术要点总结

### VLA模型架构对比
| 模型 | 参数量 | 动作生成 | 核心创新 |
|-----|:------:|:--------:|:---------|
| OpenVLA | 7B | 自回归 | 开源+双编码器 |
| DynVLA | 7B | Dynamics CoT | 动态token推理 |
| Helix | 7B+80M | 自回归 | System 1+2双系统 |
| π₀ | 3B | Flow Matching | 10步高效生成 |
| SmolVLA | <1B | 自回归 | 小模型高效率 |
| RT-2 | 55B | 自回归 | Web知识迁移 |
| ACoT-VLA | - | 动作推理 | 动作空间思维链 |
| VLA-MBPO | - | 规划+执行 | 世界模型+RL微调 |

### 动作生成机制演进
1. **Autoregressive** (RT-2, OpenVLA): 与LLM统一，逐token
2. **Flow Matching** (π₀): 10步并行，高效
3. **Diffusion** (Diffusion Policy): 多模态好，但慢
4. **Dynamics CoT** (DynVLA): 紧凑token推理

### 世界模型对比
| 模型 | 架构 | 表示方式 | 交互性 | 应用领域 |
|------|------|---------|--------|---------|
| DreamerV3 | RSSM | 隐式潜在 | ✅ | 通用RL |
| GWM | Transformer | 显式3D Gaussian | ✅ | 机器人操作 |
| Cosmos | Transformer | 隐式神经 | ✅ | 物理AI |
| V-JEPA 2 | JEPA | 隐式潜在 | ✅ | 视频理解+规划 |
| Genie 3 | 生成式 | 隐式神经 | ✅ | 交互环境生成 |

### 世界模型核心洞察
- **表示空间**: DINOv2密集特征 > VAE压缩
- **预测目标**: 结构化帧 > 密集视频 > 抽象语义
- **不确定性**: 长程推演必须考虑不确定性累积
- **架构演进**: RSSM (RNN+VAE) → Transformer → Shortcut Forcing
- **实时性**: DreamerV4实现单GPU实时交互世界模型
- **无解码器**: R2-Dreamer证明无需重建也能学习有效表示
- **显式表示**: GWM证明3D Gaussian可用于世界建模
- **数据效率**: V-JEPA 2仅需<62小时机器人数据即可零样本部署
- **交互能力**: Genie 3实现24FPS实时交互式世界生成

### 灵巧手操作技术演进
1. **RL+Sim-to-Real时代** (2018-2020)：OpenAI证明可行，但成本极高
2. **遥操作+IL时代** (2020-2023)：DexPilot降低数据收集门槛
3. **触觉感知普及** (2020-2024)：DIGIT等低成本传感器
4. **VLA大模型时代** (2023+)：RT-2等通用模型，但实时性待解

### 人形机器人产业格局
- **全球估值TOP**：Figure AI (395亿美元)、Apptronik (50亿美元)
- **国内第一梯队**：智元 (180亿)、宇树 (120亿)、银河通用 (115亿)、星海图 (100亿)
- **商业化进度**：Agility Robotics首家运营，Digit已搬运10万+箱
- **技术路线分化**：端到端VLA vs 分层模块化，电机 vs 液压
- **测试验证**：从实验室演示走向真实场景验证（10万次搬运里程碑）
- **数据采集**：遥操作（VR/同构/外骨骼）+ 自主采集 + 视频学习多路线并行

---

## 🔧 工作模式偏好

- **论文分析**: 6维度深度分析（算法细节/实验结果/创新点/对比/局限/引用）
- **产业调研**: 多维度公司分析（融资/技术/产品/商业化/团队）
- **报告整理**: 详细的markdown报告，包含表格、公式、架构图
- **知识沉淀**: 定期更新MEMORY.md，整理核心洞察
- **代码实践**: 计划从Week 3开始加入代码复现

---

## 📂 重要文件位置

```
workspace/
├── reports/                    # 论文分析报告（39篇）
│   ├── dysl-vla-arxiv-2602.22896.md
│   ├── pi0-fast-arxiv-2501.09747.md
│   ├── wmpc-multimodal-adaptation-report.md
│   ├── robust-convex-mpc-manipulators-report.md
│   ├── humanoid-robot-companies-research-report.md
│   ├── agility-robotics-deep-dive-report.md
│   ├── dexterous-hand-manipulation-algorithms-report.md
│   ├── r2-dreamer-iclr-2026-report.md
│   ├── pi-star-06-vla-rl-report.md
│   ├── dreamerv4-world-model-report.md
│   ├── robot-data-collection-methods-report.md
│   ├── robot-testing-validation-report.md
│   ├── rssm-recurrent-state-space-model-report.md
│   ├── unipi-neurips-2023-report.md              # Week 4 收官
│   ├── robocrafter-qa-soft-robot-design-report.md # Week 4 收官
│   └── robodreamer-icml-2024-report.md            # Week 4 最终收官
├── memory/                     # 每日学习日志
├── USER.md                     # 用户档案（旭哥）
├── SOUL.md                     # 我的角色设定
├── AGENTS.md                   # 工作指南
└── MEMORY.md                   # 本文件
```

---

## 🔄 更新记录

| 日期 | 更新内容 |
|-----|---------|
| 2026-03-19 | 创建 MEMORY.md，整理VLA学习项目与12篇论文 |
| 2026-04-08 | 更新学习进度至第25天，新增8篇论文 (GWM, Cosmos, V-JEPA 2, Genie 3等) |
| 2026-03-21 | 完成 SayCan + Inner Monologue 深度分析，更新论文列表至14篇 |
| 2026-03-22 | 大规模更新：新增π₀-FAST、DySL-VLA、WMPC等VLA论文；完成18家人形机器人公司调研；生成Agility深度报告；完成灵巧手操作算法综述（5篇核心论文）；总报告数达22篇 |
| 2026-03-24 | 新增R2-Dreamer深度分析（ICLR 2026）；新增π*₀.6深度分析（VLA+RL）；更新学习状态至Week 2基本完成 |
| 2026-03-25 | 新增DreamerV4深度分析（Shortcut Forcing+实时世界模型）；新增机器人数据采集方法调研报告；新增机器人功能验证测试方法调研报告 |
| 2026-03-26 | 新增RSSM详细介绍报告；总报告数达28篇；Week 2学习内容全部完成 |
| 2026-03-26 (晚) | 新增VLA机制性研究（ICLR Workshop）+ LingBot-VA因果世界模型；总报告数达30篇；Week 3正式启动 |
| 2026-04-13 | **Week 4 最终收官**！新增UniPi + RoboCrafter-QA + RoboDreamer；总报告数达**39篇**；VLA & 世界模型4周速成计划圆满完成 |

---

### Week 3 新增报告

| 序号 | 论文 | 核心贡献 | 日期 |
|:---:|------|---------|:---:|
| 31 | **ViT** | 纯Transformer图像识别，开启视觉Transformer时代 | 3/27 |
| 32 | **DiT** | 扩散模型+Transformer，adaLN条件注入，SOTA图像生成 | 3/28 |

---

*下次更新: 根据学习进度持续更新*
