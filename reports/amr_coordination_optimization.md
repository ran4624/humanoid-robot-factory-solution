# AMR 层间协同机制深度优化方案
## 边移动边操作 + 传感器共享 + 协同计算

---

**文档版本**: v2.1  
**生成日期**: 2026-04-03  
**优化重点**: 动态协同、传感器共享、计算复用

---

## 目录

1. [动态协同场景分析](#1-动态协同场景分析)
2. [边移动边操作架构](#2-边移动边操作架构)
3. [传感器共享机制](#3-传感器共享机制)
4. [协同计算优化](#4-协同计算优化)
5. [中间结果复用](#5-中间结果复用)
6. [冲突检测与解决](#6-冲突检测与解决)
7. [实现代码框架](#7-实现代码框架)

---

## 1. 动态协同场景分析

### 1.1 协同场景分类

```
┌─────────────────────────────────────────────────────────────────┐
│                    层间协同场景矩阵                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  底盘运动状态 \ 机械臂状态                                       │
│                                                                 │
│                │ 静止折叠   │ 保持姿态   │ 主动操作              │
│  ──────────────┼───────────┼───────────┼────────────           │
│  高速移动      │ 场景A     │ 场景B     │ 禁止                  │
│  ( > 1m/s )    │ 标准运输  │ 稳定持物  │ 安全限制              │
│  ──────────────┼───────────┼───────────┼────────────           │
│  中速移动      │ 场景C     │ 场景D     │ 场景E                 │
│  (0.3-1m/s)    │ 常规导航  │ 转移物品  │ 动态抓取              │
│  ──────────────┼───────────┼───────────┼────────────           │
│  低速微调      │ 场景F     │ 场景G     │ 场景H                 │
│  ( < 0.3m/s)   │ 定位准备  │ 精确定位  │ 边移动边操作          │
│  ──────────────┼───────────┼───────────┼────────────           │
│  静止          │ 场景I     │ 场景J     │ 场景K                 │
│                │ 待机     │ 预抓取    │ 标准操作              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 关键协同场景详解

#### 场景D: 中速移动 + 保持姿态 (物品转移)
```
应用场景: 抓取物品后转移到另一个位置

底盘: 中速移动 (0.5m/s)
      ↓
机械臂: 保持抓取姿态，补偿底盘运动
       • 计算底盘运动对末端的影响
       • 实时调整关节角度保持末端稳定
       • 预测到达时间，准备放置动作

协同要点:
• 底盘提供运动状态 (速度、加速度)
• 机械臂实时补偿计算
• 共享预测模型，协调减速时机
```

#### 场景E: 中速移动 + 主动操作 (动态抓取)
```
应用场景: 从移动输送带上抓取物品

底盘: 与输送带同步移动
      ↓
机械臂: 相对静止状态下执行抓取
       • 底盘跟踪输送带速度
       • 机械臂感知相对静止
       • 执行标准抓取动作

协同要点:
• 底盘跟踪控制 (视觉伺服)
• 机械臂感知相对运动补偿
• 时序协调：同步 → 抓取 → 撤离
```

#### 场景H: 低速微调 + 主动操作 (边移动边操作)
```
应用场景: 货架间狭窄空间，边调整位置边抓取

底盘: 低速微调位置 (0.1m/s)
      ↓
机械臂: 同时执行抓取/放置
       • 底盘微调扩大工作空间
       • 机械臂同时接近目标
       • 协调停止时机

协同要点:
• 运动学耦合计算
• 共享视觉感知
• 联合优化停止点
```

---

## 2. 边移动边操作架构

### 2.1 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│               边移动边操作协同架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 协同规划层 (Co-Planner)                  │   │
│  │                                                         │   │
│  │  输入: 任务目标 + 约束条件                                │   │
│  │       │                                                 │   │
│  │       ▼                                                 │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │           联合优化求解器                          │   │   │
│  │  │                                                 │   │   │
│  │  │  优化目标: min(任务时间 + 能量消耗 + 风险)       │   │   │
│  │  │                                                 │   │   │
│  │  │  决策变量:                                      │   │   │
│  │  │    • 底盘路径 [x(t), y(t), θ(t)]               │   │   │
│  │  │    • 机械臂轨迹 [q1(t), ..., q6(t)]            │   │   │
│  │  │    • 速度曲线 v(t)                             │   │   │
│  │  │                                                 │   │   │
│  │  │  约束条件:                                      │   │   │
│  │  │    • 运动学约束                                 │   │   │
│  │  │    • 动力学约束                                 │   │   │
│  │  │    • 碰撞避免                                   │   │   │
│  │  │    • 稳定性约束                                 │   │   │
│  │  │                                                 │   │   │
│  │  │  求解方法: MPC (模型预测控制)                    │   │   │
│  │  │            预测窗口: 2-3秒                       │   │   │
│  │  │            更新频率: 20Hz                        │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │       │                                                 │   │
│  │       ▼                                                 │   │
│  │  输出: 协调的底盘+机械臂轨迹                              │   │
│  │                                                         │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│           ┌────────────────┼────────────────┐                  │
│           │                │                │                  │
│           ▼                ▼                ▼                  │
│  ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐      │
│  │   底盘执行层     │ │  协同控制层  │ │   机械臂执行层   │      │
│  │  (V-JEPA 2)     │ │             │ │  (π*₀.6 VLA)    │      │
│  │                 │ │             │ │                 │      │
│  │ 轨迹跟踪控制    │◄├─ 运动补偿 ──┤►│ 运动补偿控制    │      │
│  │ 20Hz           │ │             │ │ 100Hz          │      │
│  └─────────────────┘ └─────────────┘ └─────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 运动学耦合模型

```python
class CoupledKinematics:
    """
    底盘-机械臂耦合运动学模型
    用于边移动边操作的协调控制
    """
    
    def __init__(self):
        # 底盘参数
        self.base_dofs = 3  # x, y, theta
        
        # 机械臂参数
        self.arm_dofs = 6   # 6关节
        
        # 总自由度
        self.total_dofs = self.base_dofs + self.arm_dofs
        
    def forward_kinematics(self, base_pose, arm_joints):
        """
        计算机械臂末端在世界坐标系中的位姿
        
        Args:
            base_pose: [x, y, theta] 底盘位姿
            arm_joints: [q1, ..., q6] 机械臂关节角
            
        Returns:
            tcp_pose_world: 末端在世界坐标系中的位姿
        """
        # 1. 机械臂正运动学 (相对于底盘基座)
        T_base_to_tcp = self.arm_fk(arm_joints)
        
        # 2. 底盘到世界的变换
        T_world_to_base = self.pose_to_transform(base_pose)
        
        # 3. 组合变换
        T_world_to_tcp = T_world_to_base @ T_base_to_tcp
        
        return self.transform_to_pose(T_world_to_tcp)
    
    def inverse_kinematics(self, target_tcp_world, base_pose, current_joints):
        """
        给定目标末端位姿(世界坐标系)和底盘位姿，求解机械臂关节角
        
        Args:
            target_tcp_world: 目标末端位姿 (世界坐标系)
            base_pose: 当前/规划底盘位姿
            current_joints: 当前关节角 (用于选择解)
            
        Returns:
            arm_joints: 机械臂关节角
            base_adjustment: 底盘调整建议 (如果需要)
        """
        # 1. 世界坐标系 -> 底盘坐标系
        T_world_to_base = self.pose_to_transform(base_pose)
        T_base_to_world = np.linalg.inv(T_world_to_base)
        
        target_tcp_base = T_base_to_world @ self.pose_to_transform(target_tcp_world)
        
        # 2. 机械臂逆运动学
        ik_solutions = self.arm_ik(target_tcp_base)
        
        # 3. 选择最优解 (最接近当前姿态)
        best_solution = self.select_best_ik_solution(
            ik_solutions, 
            current_joints
        )
        
        # 4. 检查解的质量
        if not self.is_solution_valid(best_solution):
            # 建议底盘调整
            base_adjustment = self.suggest_base_adjustment(
                target_tcp_world,
                base_pose
            )
            return None, base_adjustment
        
        return best_solution, None
    
    def compute_jacobian(self, base_pose, arm_joints):
        """
        计算耦合雅可比矩阵
        
        J_total = [J_base, J_arm]
        
        其中:
        - J_base: 底盘运动对末端的影响 (6x3)
        - J_arm: 机械臂关节运动对末端的影响 (6x6)
        """
        # 机械臂雅可比 (相对于底盘基座)
        J_arm = self.arm_jacobian(arm_joints)  # 6x6
        
        # 底盘雅可比
        # 底盘运动 [vx, vy, omega] 对末端速度的影响
        T_base_to_tcp = self.arm_fk(arm_joints)
        p_tcp_base = T_base_to_tcp[:3, 3]  # TCP在底盘坐标系中的位置
        
        J_base = np.zeros((6, 3))
        # 线速度部分
        J_base[0:3, 0] = [1, 0, 0]  # vx
        J_base[0:3, 1] = [0, 1, 0]  # vy
        J_base[0:3, 2] = [-p_tcp_base[1], p_tcp_base[0], 0]  # omega产生的线速度
        # 角速度部分
        J_base[3:6, 2] = [0, 0, 1]  # omega
        
        # 转换到世界坐标系
        R_world_to_base = self.pose_to_rotation_matrix(base_pose)
        J_base[0:3, :] = R_world_to_base @ J_base[0:3, :]
        J_arm[0:3, :] = R_world_to_base @ J_arm[0:3, :]
        
        # 组合雅可比
        J_total = np.hstack([J_base, J_arm])  # 6x9
        
        return J_total, J_base, J_arm
    
    def compute_motion_compensation(self, base_velocity, arm_joints):
        """
        计算底盘运动补偿
        
        当底盘移动时，机械臂需要如何运动才能保持末端静止
        
        Args:
            base_velocity: [vx, vy, omega] 底盘速度
            arm_joints: 当前机械臂关节角
            
        Returns:
            arm_compensation: 机械臂补偿速度
        """
        _, J_base, J_arm = self.compute_jacobian(
            base_pose=None,  # 不需要，雅可比已经在世界坐标系
            arm_joints=arm_joints
        )
        
        # 末端速度 = J_base @ base_velocity + J_arm @ arm_velocity
        # 令末端速度 = 0，求解 arm_velocity
        # J_arm @ arm_velocity = -J_base @ base_velocity
        
        target_tcp_velocity = np.zeros(6)
        rhs = target_tcp_velocity - J_base @ base_velocity
        
        # 使用伪逆求解
        arm_compensation = np.linalg.pinv(J_arm) @ rhs
        
        return arm_compensation
```

### 2.3 边移动边操作控制律

```python
class SimultaneousMoveAndManipulate:
    """
    边移动边操作控制器
    实现底盘和机械臂的协调控制
    """
    
    def __init__(self):
        self.kinematics = CoupledKinematics()
        
        # 控制器参数
        self.base_controller = VJEPA2BaseController()
        self.arm_controller = PiZeroArmController()
        
        # 协同参数
        self.coordination_mode = "DECOUPLED"  # DECOUPLED, COUPLED, HYBRID
        
    def control_loop(self, task_goal):
        """
        边移动边操作主控制循环
        """
        while not task_completed:
            # 1. 获取当前状态
            state = self.get_full_state()
            
            # 2. 根据协同模式选择控制策略
            if self.coordination_mode == "DECOUPLED":
                # 解耦模式: 底盘和机械臂独立控制
                base_cmd, arm_cmd = self.decoupled_control(state, task_goal)
                
            elif self.coordination_mode == "COUPLED":
                # 耦合模式: 联合优化控制
                base_cmd, arm_cmd = self.coupled_control(state, task_goal)
                
            elif self.coordination_mode == "HYBRID":
                # 混合模式: 根据任务阶段切换
                base_cmd, arm_cmd = self.hybrid_control(state, task_goal)
            
            # 3. 安全检查
            if not self.safety_check(base_cmd, arm_cmd, state):
                base_cmd, arm_cmd = self.emergency_stop()
            
            # 4. 执行控制指令
            self.execute_commands(base_cmd, arm_cmd)
            
            # 5. 更新任务状态
            self.update_task_progress()
    
    def coupled_control(self, state, task_goal):
        """
        耦合控制: 联合优化底盘和机械臂动作
        
        使用 MPC 求解最优的底盘+机械臂轨迹
        """
        # 构建 MPC 问题
        mpc_problem = self.build_mpc_problem(state, task_goal)
        
        # 求解
        solution = self.mpc_solver.solve(mpc_problem)
        
        # 提取控制指令
        base_cmd = solution['base_velocity'][0]  # 第一步
        arm_cmd = solution['arm_velocity'][0]
        
        return base_cmd, arm_cmd
    
    def build_mpc_problem(self, state, task_goal):
        """
        构建模型预测控制问题
        """
        N = 20  # 预测步数
        dt = 0.05  # 时间步长 (20Hz)
        
        # 决策变量
        # X = [base_pose(3), arm_joints(6)] @ each timestep
        # U = [base_velocity(3), arm_velocity(6)] @ each timestep
        
        problem = {
            'horizon': N,
            'dt': dt,
            'initial_state': state,
            'target': task_goal,
            
            # 代价函数
            'cost': {
                'terminal': self.terminal_cost,
                'running': self.running_cost,
            },
            
            # 约束
            'constraints': {
                'dynamics': self.coupled_dynamics,
                'joint_limits': self.arm_joint_limits,
                'velocity_limits': self.velocity_limits,
                'collision_avoidance': self.collision_constraints,
                'stability': self.stability_constraints,
            }
        }
        
        return problem
    
    def coupled_dynamics(self, x, u):
        """
        耦合动力学方程
        
        x = [base_x, base_y, base_theta, q1, q2, q3, q4, q5, q6]
        u = [base_vx, base_vy, base_omega, dq1, dq2, dq3, dq4, dq5, dq6]
        """
        base_pose = x[0:3]
        arm_joints = x[3:9]
        
        base_velocity = u[0:3]
        arm_velocity = u[3:9]
        
        # 底盘运动学
        base_pose_dot = self.base_kinematics(base_pose, base_velocity)
        
        # 机械臂运动学
        arm_joints_dot = arm_velocity
        
        x_dot = np.concatenate([base_pose_dot, arm_joints_dot])
        
        return x_dot
    
    def hybrid_control(self, state, task_goal):
        """
        混合控制策略
        根据任务阶段自动切换控制模式
        """
        phase = self.determine_task_phase(state, task_goal)
        
        if phase == "APPROACH":
            # 接近阶段: 底盘主导，机械臂准备
            self.coordination_mode = "DECOUPLED"
            base_cmd = self.base_controller.plan_to_goal(
                state['base_pose'], 
                task_goal['approach_pose']
            )
            arm_cmd = self.arm_controller.move_to_ready_pose()
            
        elif phase == "FINE_POSITIONING":
            # 精确定位: 耦合控制，协同优化
            self.coordination_mode = "COUPLED"
            base_cmd, arm_cmd = self.coupled_control(state, task_goal)
            
        elif phase == "MANIPULATION":
            # 操作阶段: 底盘静止或微动，机械臂主导
            if self.can_manipulate_while_moving(state):
                self.coordination_mode = "COUPLED"
                base_cmd, arm_cmd = self.coupled_control(state, task_goal)
            else:
                self.coordination_mode = "DECOUPLED"
                base_cmd = np.zeros(3)  # 底盘静止
                arm_cmd = self.arm_controller.execute_task(task_goal)
                
        elif phase == "TRANSFER":
            # 转移阶段: 底盘移动，机械臂补偿保持
            self.coordination_mode = "DECOUPLED"
            base_cmd = self.base_controller.plan_to_goal(
                state['base_pose'],
                task_goal['next_location']
            )
            arm_cmd = self.compute_motion_compensation(
                base_cmd,
                state['arm_joints']
            )
        
        return base_cmd, arm_cmd
```

---

## 3. 传感器共享机制

### 3.1 共享传感器架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    共享传感器架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  共享传感器层                             │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │  前视相机    │  │  激光雷达    │  │  IMU        │     │   │
│  │  │ (共享主摄)  │  │ (共享点云)  │  │ (共享姿态)  │     │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │   │
│  │         │                │                │             │   │
│  │         └────────────────┼────────────────┘             │   │
│  │                          │                             │   │
│  │                          ▼                             │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              传感器融合中心                       │   │   │
│  │  │                                                 │   │   │
│  │  │  • 时间同步 (硬件/软件同步)                      │   │   │
│  │  │  • 空间标定 (外参标定)                          │   │   │
│  │  │  • 数据对齐 (时间戳对齐)                        │   │   │
│  │  │  • 质量评估 (异常检测)                          │   │   │
│  │  │                                                 │   │   │
│  │  │  输出: 统一的感知表示                            │   │   │
│  │  │       • 全局地图                                 │   │   │
│  │  │       • 局部点云                                 │   │   │
│  │  │       • 语义分割                                 │   │   │
│  │  │       • 动态物体跟踪                             │   │   │
│  │  └────────────────────────┬────────────────────────┘   │   │
│  │                           │                            │   │
│  └───────────────────────────┼────────────────────────────┘   │
│                              │                                 │
│           ┌──────────────────┼──────────────────┐              │
│           │                  │                  │              │
│           ▼                  ▼                  ▼              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │   底盘使用       │ │   机械臂使用     │ │   协同使用       │  │
│  │                 │ │                 │ │                 │  │
│  │ • 全局导航      │ │ • 局部操作      │ │ • 联合感知      │  │
│  │ • 障碍物检测    │ │ • 精细抓取      │ │ • 手眼协调      │  │
│  │ • 定位          │ │ • 力控          │ │ • 碰撞检测      │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 传感器共享实现

```python
class SharedSensorHub:
    """
    共享传感器中心
    管理所有传感器的采集、融合和分发
    """
    
    def __init__(self):
        # 传感器实例
        self.sensors = {
            'front_camera': FrontCamera(),
            'lidar': Lidar2D(),
            'imu': IMU(),
            'wrist_camera': WristCamera(),
        }
        
        # 订阅者管理
        self.subscribers = {
            'base_nav': [],
            'arm_manip': [],
            'coordinator': [],
        }
        
        # 融合算法
        self.fusion = SensorFusion()
        
        # 数据缓存
        self.cache = SensorDataCache()
        
    def initialize(self):
        """
        初始化传感器和标定
        """
        # 1. 加载标定参数
        self.extrinsics = self.load_calibration()
        
        # 2. 时间同步
        self.sync = TimeSynchronizer(
            sensors=self.sensors,
            tolerance_ms=10
        )
        
        # 3. 启动采集线程
        for sensor in self.sensors.values():
            sensor.start()
    
    def register_subscriber(self, module, sensor_types, callback):
        """
        注册传感器数据订阅者
        
        Args:
            module: 'base_nav', 'arm_manip', 或 'coordinator'
            sensor_types: 需要的传感器类型列表
            callback: 数据回调函数
        """
        self.subscribers[module].append({
            'types': sensor_types,
            'callback': callback,
            'frequency': self.get_required_frequency(module)
        })
    
    def process_and_distribute(self):
        """
        处理传感器数据并分发给订阅者
        """
        # 1. 采集同步数据
        synced_data = self.sync.get_synced_data()
        
        # 2. 数据融合
        fused_data = self.fusion.process(synced_data)
        
        # 3. 缓存
        self.cache.store(fused_data)
        
        # 4. 分发给订阅者
        for module, subs in self.subscribers.items():
            for sub in subs:
                # 提取订阅者需要的数据
                data = self.extract_data(fused_data, sub['types'])
                
                # 降采样到订阅者需要的频率
                data = self.downsample(data, sub['frequency'])
                
                # 调用回调
                sub['callback'](data)
    
    def get_transform(self, source_frame, target_frame, timestamp):
        """
        获取两个坐标系之间的变换
        
        用于将传感器数据转换到需要的坐标系
        """
        # 查找或插值获取变换
        transform = self.cache.lookup_transform(
            source_frame,
            target_frame,
            timestamp
        )
        return transform


class SensorFusion:
    """
    多传感器融合算法
    """
    
    def __init__(self):
        # 视觉-LiDAR融合
        self.camera_lidar_fusion = CameraLidarFusion()
        
        # 视觉-IMU融合
        self.visual_inertial_odometry = VIO()
        
        # 语义融合
        self.semantic_fusion = SemanticFusion()
        
    def process(self, synced_data):
        """
        融合多传感器数据
        """
        # 1. 视觉-LiDAR融合 (稠密深度)
        dense_depth = self.camera_lidar_fusion.fuse(
            synced_data['front_camera'],
            synced_data['lidar']
        )
        
        # 2. 视觉-惯性里程计
        odometry = self.visual_inertial_odometry.estimate(
            synced_data['front_camera'],
            synced_data['imu']
        )
        
        # 3. 语义分割融合
        semantics = self.semantic_fusion.segment(
            synced_data['front_camera'],
            dense_depth
        )
        
        return {
            'dense_depth': dense_depth,
            'odometry': odometry,
            'semantics': semantics,
            'raw_data': synced_data
        }
```

### 3.3 具体共享场景

#### 场景1: 前视相机共享

```python
class FrontCameraSharing:
    """
    前视相机共享使用场景
    """
    
    def __init__(self, shared_hub):
        self.hub = shared_hub
        
        # 注册订阅者
        self.hub.register_subscriber(
            module='base_nav',
            sensor_types=['front_camera'],
            callback=self.base_nav_callback
        )
        
        self.hub.register_subscriber(
            module='arm_manip',
            sensor_types=['front_camera'],
            callback=self.arm_manip_callback
        )
        
    def base_nav_callback(self, data):
        """
        底盘导航使用
        
        用途:
        - V-JEPA 2 视频输入
        - 障碍物检测
        - 场景理解
        """
        # 降采样到适合 V-JEPA 2 的分辨率
        frame = cv2.resize(data['image'], (640, 360))
        
        # 添加到视频缓冲区
        self.base_video_buffer.append(frame)
        
        # 触发 V-JEPA 2 推理
        if len(self.base_video_buffer) >= 8:
            self.trigger_vjepa_inference()
    
    def arm_manip_callback(self, data):
        """
        机械臂操作使用
        
        用途:
        - 大范围场景感知
        - 目标定位 (当腕部相机看不到时)
        - 路径规划辅助
        """
        # 转换到机械臂坐标系
        transform = self.hub.get_transform(
            'front_camera_optical',
            'arm_base_link',
            data['timestamp']
        )
        
        # 投影到机械臂工作空间
        arm_view = self.project_to_arm_workspace(
            data['image'],
            data['depth'],
            transform
        )
        
        # 提供给 VLA 作为辅助输入
        self.arm_context_view = arm_view
```

#### 场景2: LiDAR 点云共享

```python
class LidarSharing:
    """
    LiDAR 点云共享使用
    """
    
    def process_shared_lidar(self, point_cloud):
        """
        处理共享的 LiDAR 数据
        """
        # 1. 底盘使用: 导航和定位
        base_usage = self.process_for_base(point_cloud)
        
        # 2. 机械臂使用: 碰撞检测
        arm_usage = self.process_for_arm(point_cloud)
        
        # 3. 协同使用: 联合避障
        coordinated_usage = self.process_for_coordination(point_cloud)
        
        return {
            'base': base_usage,
            'arm': arm_usage,
            'coordinated': coordinated_usage
        }
    
    def process_for_base(self, point_cloud):
        """
        为底盘导航处理点云
        """
        # 地面分割
        ground, obstacles = self.segment_ground(point_cloud)
        
        # 2D投影用于导航
        costmap = self.project_to_2d(obstacles)
        
        return {
            'costmap': costmap,
            'obstacles_2d': obstacles
        }
    
    def process_for_arm(self, point_cloud):
        """
        为机械臂操作处理点云
        """
        # 提取机械臂工作空间内的点云
        arm_workspace_cloud = self.extract_workspace_points(
            point_cloud,
            workspace_bounds=self.arm_workspace
        )
        
        # 构建局部碰撞地图
        arm_collision_map = self.build_collision_map(arm_workspace_cloud)
        
        return {
            'workspace_cloud': arm_workspace_cloud,
            'collision_map': arm_collision_map
        }
    
    def process_for_coordination(self, point_cloud):
        """
        为协同控制处理点云
        """
        # 构建完整的 3D 碰撞地图
        full_collision_map = self.build_3d_collision_map(point_cloud)
        
        # 预测动态障碍物
        dynamic_obstacles = self.predict_dynamics(point_cloud)
        
        return {
            'full_collision_map': full_collision_map,
            'dynamic_obstacles': dynamic_obstacles
        }
```

---

## 4. 协同计算优化

### 4.1 计算资源共享

```
┌─────────────────────────────────────────────────────────────────┐
│                    协同计算架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  共享计算资源池                            │   │
│  │                                                         │   │
│  │  ┌─────────────────┐    ┌─────────────────┐            │   │
│  │  │   GPU 0 (Orin)  │    │   GPU 1 (Orin)  │            │   │
│  │  │  275 TOPS       │    │  275 TOPS       │            │   │
│  │  │                 │    │                 │            │   │
│  │  │  机械臂 VLA     │    │  V-JEPA 2       │            │   │
│  │  │  推理 + 学习    │    │  编码 + 预测    │            │   │
│  │  └─────────────────┘    └─────────────────┘            │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              CPU 共享资源                        │   │   │
│  │  │  • 传感器预处理    • 数据缓存    • 通信处理      │   │   │
│  │  │  • 坐标变换        • 任务调度    • 日志记录      │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│           ┌────────────────┼────────────────┐                   │
│           │                │                │                   │
│           ▼                ▼                ▼                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   计算任务调度    │ │   中间结果缓存    │ │   负载均衡       │   │
│  │                 │ │                 │ │                 │   │
│  │ • 优先级管理    │ │ • 特征复用      │ │ • 动态分配      │   │
│  │ • 资源分配      │ │ • 结果共享      │ │ • 任务迁移      │   │
│  │ • 冲突解决      │ │ • 增量计算      │ │ • 故障恢复      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 动态负载均衡

```python
class ComputeResourceManager:
    """
    计算资源动态管理器
    根据任务负载动态分配计算资源
    """
    
    def __init__(self):
        self.gpus = [GPU(0), GPU(1)]
        self.cpu_cores = 8
        
        # 任务队列
        self.task_queues = {
            'arm_inference': PriorityQueue(),
            'base_inference': PriorityQueue(),
            'sensor_processing': PriorityQueue(),
            'learning': PriorityQueue(),
        }
        
        # 资源监控
        self.monitor = ResourceMonitor()
        
    def schedule_task(self, task):
        """
        调度任务到合适的计算资源
        """
        # 分析任务需求
        requirements = self.analyze_requirements(task)
        
        # 获取当前资源状态
        resource_status = self.monitor.get_status()
        
        # 选择最佳资源
        if requirements['type'] == 'inference':
            # 推理任务优先使用 GPU
            gpu_id = self.select_gpu(resource_status, requirements)
            self.assign_to_gpu(task, gpu_id)
            
        elif requirements['type'] == 'preprocessing':
            # 预处理任务使用 CPU
            cores = self.select_cpu_cores(resource_status, requirements)
            self.assign_to_cpu(task, cores)
            
        elif requirements['type'] == 'learning':
            # 学习任务使用空闲 GPU
            gpu_id = self.find_idle_gpu(resource_status)
            if gpu_id is not None:
                self.assign_to_gpu(task, gpu_id)
            else:
                # 排队等待
                self.task_queues['learning'].put(task)
    
    def select_gpu(self, resource_status, requirements):
        """
        选择最佳 GPU
        """
        # 考虑因素:
        # 1. 当前负载
        # 2. 内存使用
        # 3. 任务亲和性 (相似任务放同一GPU)
        
        scores = []
        for i, gpu in enumerate(self.gpus):
            score = 0
            
            # 负载越低越好
            score += (100 - gpu.load) * 0.4
            
            # 内存越充足越好
            score += (gpu.memory_free / gpu.memory_total) * 100 * 0.3
            
            # 任务亲和性
            if gpu.last_task_type == requirements['task_type']:
                score += 30 * 0.3
            
            scores.append((i, score))
        
        # 选择得分最高的 GPU
        best_gpu = max(scores, key=lambda x: x[1])[0]
        return best_gpu
```

---

## 5. 中间结果复用

### 5.1 特征复用机制

```python
class FeatureCache:
    """
    特征缓存系统
    缓存和复用中间计算结果
    """
    
    def __init__(self):
        # 视觉特征缓存
        self.visual_cache = LRUCache(maxsize=1000)
        
        # 场景理解缓存
        self.scene_cache = TTLCache(maxsize=100, ttl=5.0)
        
        # 预测结果缓存
        self.prediction_cache = TTLCache(maxsize=50, ttl=1.0)
        
    def get_visual_features(self, image, model_name):
        """
        获取视觉特征，优先从缓存读取
        """
        # 计算图像哈希
        image_hash = self.compute_image_hash(image)
        cache_key = f"{model_name}_{image_hash}"
        
        # 检查缓存
        if cache_key in self.visual_cache:
            return self.visual_cache[cache_key]
        
        # 计算特征
        features = self.compute_visual_features(image, model_name)
        
        # 存入缓存
        self.visual_cache[cache_key] = features
        
        return features
    
    def share_features_between_layers(self):
        """
        在层间共享特征
        """
        # 场景: 底盘 V-JEPA 2 编码的视频特征可以复用给机械臂
        
        # 1. V-JEPA 2 编码当前场景
        vjepa_features = self.vjepa_encoder(video_clip)
        
        # 2. 投影到 VLA 视觉特征空间
        shared_features = self.feature_projector(
            vjepa_features,
            source='vjepa',
            target='vla'
        )
        
        # 3. 机械臂 VLA 复用这些特征
        # 减少重复编码计算
        return shared_features


class FeatureProjector(nn.Module):
    """
    特征投影网络
    将一种特征空间映射到另一种
    """
    
    def __init__(self, source_dim, target_dim):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(source_dim, 512),
            nn.ReLU(),
            nn.Linear(512, target_dim)
        )
        
    def forward(self, source_features):
        return self.projector(source_features)
```

### 5.2 预测结果复用

```python
class PredictionReuse:
    """
    预测结果复用
    V-JEPA 2 的预测结果可用于多个目的
    """
    
    def __init__(self, vjepa_model):
        self.vjepa = vjepa_model
        
    def multi_use_prediction(self, current_state, action_plan):
        """
        一次预测，多处使用
        """
        # 执行一次世界模型预测
        predicted_states = self.vjepa.predict_future(
            current_state,
            action_plan
        )
        
        # 1. 底盘使用: 碰撞检测
        collision_risks = self.assess_collision_risks(predicted_states)
        
        # 2. 机械臂使用: 工作空间可达性检查
        workspace_validity = self.check_workspace_validity(predicted_states)
        
        # 3. 协同使用: 联合稳定性评估
        stability_scores = self.assess_stability(predicted_states)
        
        # 4. 任务规划: 预测完成时间
        estimated_completion = self.estimate_completion_time(predicted_states)
        
        return {
            'predicted_states': predicted_states,
            'collision_risks': collision_risks,
            'workspace_validity': workspace_validity,
            'stability_scores': stability_scores,
            'estimated_completion': estimated_completion
        }
```

---

## 6. 冲突检测与解决

### 6.1 冲突类型

```
┌─────────────────────────────────────────────────────────────────┐
│                    层间冲突类型                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 运动学冲突                                                   │
│     • 机械臂运动超出底盘稳定范围                                 │
│     • 底盘移动导致机械臂工作空间变化                             │
│     • 联合运动导致奇异点                                         │
│                                                                 │
│  2. 感知冲突                                                     │
│     • 传感器数据不一致                                           │
│     • 坐标变换误差累积                                           │
│     • 时间同步偏差                                               │
│                                                                 │
│  3. 资源冲突                                                     │
│     • 计算资源竞争                                               │
│     • 通信带宽不足                                               │
│     • 传感器访问冲突                                             │
│                                                                 │
│  4. 任务冲突                                                     │
│     • 底盘和机械臂目标矛盾                                       │
│     • 优先级冲突                                                 │
│     • 时序不协调                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 冲突检测与解决

```python
class ConflictResolver:
    """
    层间冲突检测与解决
    """
    
    def __init__(self):
        self.kinematics = CoupledKinematics()
        self.safety_limits = SafetyLimits()
        
    def detect_conflicts(self, base_plan, arm_plan):
        """
        检测底盘和机械臂计划之间的冲突
        """
        conflicts = []
        
        # 1. 检查运动学冲突
        kin_conflict = self.check_kinematic_conflicts(base_plan, arm_plan)
        if kin_conflict:
            conflicts.append(kin_conflict)
        
        # 2. 检查稳定性冲突
        stab_conflict = self.check_stability_conflicts(base_plan, arm_plan)
        if stab_conflict:
            conflicts.append(stab_conflict)
        
        # 3. 检查碰撞冲突
        coll_conflict = self.check_collision_conflicts(base_plan, arm_plan)
        if coll_conflict:
            conflicts.append(coll_conflict)
        
        return conflicts
    
    def check_kinematic_conflicts(self, base_plan, arm_plan):
        """
        检查运动学冲突
        """
        for t in range(len(base_plan['trajectory'])):
            base_pose = base_plan['trajectory'][t]
            arm_joints = arm_plan['trajectory'][t]
            
            # 检查机械臂是否在工作空间内
            tcp_pose = self.kinematics.forward_kinematics(base_pose, arm_joints)
            
            if not self.is_in_workspace(tcp_pose):
                return {
                    'type': 'KINEMATIC',
                    'time': t,
                    'description': 'TCP out of workspace',
                    'severity': 'HIGH'
                }
            
            # 检查是否接近奇异点
            if self.is_near_singularity(base_pose, arm_joints):
                return {
                    'type': 'KINEMATIC',
                    'time': t,
                    'description': 'Near singularity',
                    'severity': 'MEDIUM'
                }
        
        return None
    
    def resolve_conflicts(self, conflicts, base_plan, arm_plan):
        """
        解决检测到的冲突
        """
        resolved_base = base_plan.copy()
        resolved_arm = arm_plan.copy()
        
        for conflict in conflicts:
            if conflict['type'] == 'KINEMATIC':
                # 运动学冲突: 调整底盘位置或机械臂姿态
                resolved_base, resolved_arm = self.resolve_kinematic_conflict(
                    conflict, resolved_base, resolved_arm
                )
                
            elif conflict['type'] == 'STABILITY':
                # 稳定性冲突: 减速或暂停
                resolved_base, resolved_arm = self.resolve_stability_conflict(
                    conflict, resolved_base, resolved_arm
                )
                
            elif conflict['type'] == 'COLLISION':
                # 碰撞冲突: 重新规划
                resolved_base, resolved_arm = self.resolve_collision_conflict(
                    conflict, resolved_base, resolved_arm
                )
        
        return resolved_base, resolved_arm
    
    def resolve_kinematic_conflict(self, conflict, base_plan, arm_plan):
        """
        解决运动学冲突
        
        策略: 优先调整底盘，其次调整机械臂
        """
        t = conflict['time']
        
        # 尝试调整底盘位置
        adjusted_base = self.adjust_base_position(
            base_plan,
            t,
            arm_plan['trajectory'][t]
        )
        
        # 检查是否解决
        if self.check_kinematic_conflicts(adjusted_base, arm_plan) is None:
            return adjusted_base, arm_plan
        
        # 如果底盘调整不够，调整机械臂
        adjusted_arm = self.adjust_arm_pose(
            arm_plan,
            t,
            base_plan['trajectory'][t]
        )
        
        return base_plan, adjusted_arm
```

---

## 7. 实现代码框架

### 7.1 完整协同系统代码结构

```python
# main_coordinator.py
class AMRCoordinatedSystem:
    """
    AMR 分层协同系统主控制器
    """
    
    def __init__(self):
        # 传感器共享中心
        self.sensor_hub = SharedSensorHub()
        
        # 计算资源管理
        self.compute_manager = ComputeResourceManager()
        
        # 特征缓存
        self.feature_cache = FeatureCache()
        
        # 层控制器
        self.base_controller = CoordinatedBaseController(
            sensor_hub=self.sensor_hub,
            feature_cache=self.feature_cache
        )
        
        self.arm_controller = CoordinatedArmController(
            sensor_hub=self.sensor_hub,
            feature_cache=self.feature_cache
        )
        
        # 协同规划器
        self.co_planner = CoordinatedPlanner(
            base_controller=self.base_controller,
            arm_controller=self.arm_controller
        )
        
        # 冲突解决器
        self.conflict_resolver = ConflictResolver()
        
        # 任务管理器
        self.task_manager = TaskManager()
        
    def run(self):
        """
        主运行循环
        """
        # 初始化
        self.initialize()
        
        while self.running:
            # 1. 获取任务
            task = self.task_manager.get_next_task()
            
            # 2. 协同规划
            base_plan, arm_plan = self.co_planner.plan(task)
            
            # 3. 冲突检测
            conflicts = self.conflict_resolver.detect_conflicts(
                base_plan, arm_plan
            )
            
            # 4. 冲突解决
            if conflicts:
                base_plan, arm_plan = self.conflict_resolver.resolve_conflicts(
                    conflicts, base_plan, arm_plan
                )
            
            # 5. 执行
            self.execute_coordinated(base_plan, arm_plan)
            
            # 6. 监控和反馈
            self.monitor_and_adapt()
    
    def execute_coordinated(self, base_plan, arm_plan):
        """
        协同执行底盘和机械臂计划
        """
        # 创建同步执行器
        executor = SynchronizedExecutor(
            base_controller=self.base_controller,
            arm_controller=self.arm_controller,
            sync_frequency=100  # Hz
        )
        
        # 执行
        executor.execute(base_plan, arm_plan)
```

---

**文档结束**

*层间协同机制深度优化：边移动边操作 + 传感器共享 + 协同计算*
