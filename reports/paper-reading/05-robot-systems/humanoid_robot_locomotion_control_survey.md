# 人形机器人运动控制通用控制器算法调研报告

## 一、概述

人形机器人运动控制是一个多层次、多时间尺度的复杂问题，涉及从毫秒级的关节力控到秒级的步态规划。当前主流的控制器架构通常采用分层设计，将问题分解为不同的抽象层次。

---

## 二、主流控制架构

### 2.1 分层控制架构 (Hierarchical Control)

```
┌─────────────────────────────────────────────────────────────┐
│  高层规划 (High-Level Planning)                              │
│  - 路径规划、导航、任务规划                                   │
│  - 时间尺度: 0.1-1s                                          │
├─────────────────────────────────────────────────────────────┤
│  步态规划 (Gait Planning)                                    │
│  - 步态生成、落脚点规划、质心轨迹                             │
│  - 时间尺度: 0.01-0.1s                                       │
├─────────────────────────────────────────────────────────────┤
│  全身控制 (Whole-Body Control)                               │
│  - 多任务优先级、接触力分配、运动学协调                        │
│  - 时间尺度: 1-10ms                                          │
├─────────────────────────────────────────────────────────────┤
│  关节控制 (Joint Control)                                    │
│  - 力/位置控制、阻抗控制、PD控制                              │
│  - 时间尺度: 0.1-1ms                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、核心控制算法

### 3.1 模型预测控制 (Model Predictive Control, MPC)

**原理**: 在有限时域内求解最优控制问题，通过滚动优化实现反馈控制。

**数学形式**:
```
minimize   Σ(x_k^T Q x_k + u_k^T R u_k) + x_N^T P x_N
subject to x_{k+1} = f(x_k, u_k)
           x_k ∈ X, u_k ∈ U
           g(x_k, u_k) = 0  (接触约束)
```

**特点**:
- ✅ 显式处理约束（关节限位、力摩擦锥）
- ✅ 预测未来状态，提前调整
- ✅ 自然处理多接触点
- ⚠️ 计算量大，需要实时求解QP

**代表工作**:
- MIT Cheetah 3/mini Cheetah 的凸模型预测控制 (Convex MPC)
- ETH Zürich 的 Differential Dynamic Programming (DDP) 方法
- IHMC 的 Linear MPC for walking

**应用案例**:
- Boston Dynamics Atlas 的步态控制
- Unitree H1/G1 的运动控制
- Tesla Optimus 的行走控制

---

### 3.2 全身控制 (Whole-Body Control, WBC)

**原理**: 通过任务优先级和零空间投影，协调多个控制目标。

**数学形式** (基于QP的WBC):
```
minimize   ||J_c q̈ + J̇_c q̇ - a_c||^2 + ||τ||^2
subject to M q̈ + h = S^T τ + J_c^T f
           f ∈ friction_cone
           τ_min ≤ τ ≤ τ_max
```

**零空间投影方法**:
```
τ = τ_1 + N_1 (τ_2 + N_2 (τ_3 + ...))
```

其中 N_i 是第i层任务的零空间投影矩阵。

**任务优先级示例**:
1. 质心动量跟踪 (最高优先级)
2. 摆动脚轨迹跟踪
3. 上身姿态保持
4. 关节位置正则化 (最低优先级)

**代表工作**:
- Stanford 的 Task-Space Inertia Matrix (TSIM) 方法
- LAAS-CNRS 的 Stack of Tasks (SoT) 框架
- MIT 的 Floating Base WBC

---

### 3.3 混合零动态控制 (Hybrid Zero Dynamics, HZD)

**原理**: 将周期性行走建模为混合系统，设计虚拟约束使得零动态稳定。

**核心思想**:
- 定义输出函数 y = h(q) 表示虚拟约束
- 通过反馈线性化使得 y → 0
- 分析零动态系统的稳定性

**数学形式**:
```
y = h(q) = q_a - h_d(s(q))
```

其中 q_a 是驱动关节，h_d 是期望轨迹，s 是步态相位。

**特点**:
- ✅ 保证周期性步态的指数稳定性
- ✅ 可分析证明稳定性
- ✅ 能量效率高
- ⚠️ 需要预先设计步态库
- ⚠️ 对模型误差敏感

**代表工作**:
- Jessy Grizzle (UMich) 的 HZD 理论框架
- AMBER Lab 的系列人形机器人 (AMBER 1-3, Cassie)
- Agility Robotics Digit 的控制基础

---

### 3.4 强化学习控制 (Reinforcement Learning, RL)

**原理**: 通过试错学习最优策略，无需显式系统模型。

**常用算法**:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)

**观察空间设计**:
```python
obs = [
    # 本体感知
    joint_positions,      # 关节位置
    joint_velocities,     # 关节速度
    imu_orientation,      # IMU姿态
    imu_angular_velocity, # IMU角速度
    
    # 指令
    command_velocity,     # 目标速度
    command_yaw_rate,     # 目标偏航角速度
    
    # 历史信息
    previous_actions,     # 历史动作
]
```

**动作空间设计**:
- 关节位置偏移 (PD目标位置)
- 关节力矩直接输出
- 足端位置目标 (逆运动学)

**奖励函数设计**:
```python
reward = w1 * velocity_tracking + \
         w2 * upright_penalty + \
         w3 * energy_penalty + \
         w4 * foot_clearance + \
         w5 * smoothness_reward
```

**特点**:
- ✅ 可学习复杂、非线性策略
- ✅ 适应不确定性
- ✅ 可实现敏捷运动 (跑、跳、后空翻)
- ⚠️ 需要大量仿真训练
- ⚠️ Sim-to-Real 迁移挑战
- ⚠️ 安全性、可解释性问题

**代表工作**:
- Berkeley 的 Cassie 深度强化学习控制
- ETH Zürich 的 ANYmal 强化学习控制
- 宇树 H1/G1 的强化学习步态
- 智元远征 A2 的强化学习运动控制

---

### 3.5 阻抗/导纳控制 (Impedance/Admittance Control)

**原理**: 调节机器人与环境之间的动态交互关系。

**阻抗控制** (力控):
```
F = M_d (ẍ_d - ẍ) + B_d (ẋ_d - ẋ) + K_d (x_d - x)
```

**导纳控制** (位置控):
```
ẍ = M_d^{-1} (F_ext - B_d (ẋ - ẋ_d) - K_d (x - x_d))
```

**应用场景**:
- 足地接触力控制
- 人机交互
- 未知环境适应

---

## 四、关键技术与挑战

### 4.1 接触力控制

**摩擦锥约束**:
```
|f_t| ≤ μ f_n  (库仑摩擦)
```

**零力矩点 (ZMP) 约束**:
```
p_ZMP = (m g x_c - z_c m ẍ_c) / (m g)
```

**质心角动量控制**:
```
L = I ω
Ḻ = τ_ext
```

### 4.2 状态估计

**扩展卡尔曼滤波 (EKF)**:
- 融合IMU、关节编码器、力传感器
- 估计质心位置、速度、姿态

**粒子滤波**:
- 处理非高斯噪声
- 多模态状态分布

**学习-based 状态估计**:
- 使用神经网络补偿模型误差
- 端到端学习状态估计器

### 4.3 Sim-to-Real 迁移

**领域随机化 (Domain Randomization)**:
- 随机化质量、摩擦、延迟等参数
- 提高策略鲁棒性

**系统辨识**:
- 辨识真实机器人动力学参数
- 减小仿真与现实的差距

**残差学习**:
- 学习仿真与现实之间的残差动力学
- 在线适应

---

## 五、主流人形机器人控制方案对比

| 机器人 | 控制架构 | 关键算法 | 特点 |
|:-------|:---------|:---------|:-----|
| **Atlas (Boston Dynamics)** | MPC + WBC | 凸MPC、QP-based WBC | 高动态运动、液压驱动 |
| **Digit (Agility Robotics)** | HZD + MPC | 混合零动态、线性MPC | 高效行走、电动驱动 |
| **Optimus (Tesla)** | RL + MPC | 强化学习、模型预测控制 | 端到端学习、大规模数据 |
| **Figure 01 (Figure AI)** | 神经网络 + MPC | 端到端VLA、Helix架构 | 多模态感知、任务级控制 |
| **H1/G1 (宇树)** | RL + WBC | 强化学习、全身控制 | 低成本、高敏捷 |
| **远征 A2 (智元)** | RL + MPC | 强化学习、模型预测 | 开源、模块化 |
| **GR-1 (傅利叶)** | WBC + Impedance | 全身控制、阻抗控制 | 工业级、高负载 |
| **Walker X (优必选)** | MPC + WBC | 模型预测、全身控制 | 商业化、服务场景 |

---

## 六、发展趋势

### 6.1 端到端学习
- 从感知到动作的端到端神经网络
- 减少手工设计组件
- 代表: Tesla Optimus、Figure Helix

### 6.2 多模态融合
- 视觉、力觉、本体感觉的融合
- 语义理解与运动控制结合
- 代表: VLA (Vision-Language-Action) 模型

### 6.3 世界模型 (World Model)
- 学习环境的预测模型
- 用于MPC的预测和规划
- 代表: Dreamer、RSSM

### 6.4 全身协调与操作
- 行走与操作的统一控制
- 动态稳定下的双臂操作
- 代表: 人形机器人全身操作任务

### 6.5 安全与鲁棒性
- 形式化验证的控制器
- 安全关键约束的强化学习
- 故障检测与恢复

---

## 七、推荐学习资源

### 论文
1. "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control" (Katz et al., 2018)
2. "Feedback Control of a Cassie Bipedal Robot: Walking, Standing, and Riding a Segway" (Gong et al., 2019)
3. "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning" (Rudin et al., 2022)
4. "Whole-Body MPC and Online Gait Sequence Generation for Wheeled-Legged Robots" (Villarreal et al., 2022)

### 开源代码
- **OCS2**: ETH Zürich 的MPC工具箱
- **Crocoddyl**: 基于DDP的最优控制库
- **Pinocchio**: 刚体动力学库
- **Drake**: MIT/MIT CSAIL 的机器人仿真与控制平台

### 课程
- MIT 6.832: Underactuated Robotics
- ETH Zürich: Robot Dynamics
- CMU 16-741: Mechanics of Manipulation

---

## 八、总结

人形机器人运动控制正处于快速发展期，从传统的基于模型的方法 (MPC、WBC、HZD) 向数据驱动的学习方法 (RL、VLA) 演进。未来的趋势是两者的融合：利用学习增强模型的适应性，同时保持基于模型方法的安全性和可解释性。

关键挑战包括：
1. 实时计算与复杂算法的平衡
2. Sim-to-Real 迁移的可靠性
3. 安全关键应用的形式化保证
4. 能耗与动态性能的平衡

---

*报告生成时间: 2026-03-30*
*基于公开文献与开源项目整理*
