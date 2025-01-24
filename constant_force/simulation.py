import numpy as np
import matplotlib.pyplot as plt
import random

T = 1             # 仿真时间 (秒)
dt = 0.0001       # 时间步长 (秒)
time = np.arange(0, T, dt)

# 参数设定
C = 1.2             # 材料常数
d = 10              # 钻头直径 (mm)
F_target = 200      # 目标切削力 (N)

def f(Kp, Ki, Kd):

    
    # 初始条件
    v_f = 0             # 初始进给速度
    F_actual = 0        # 初始切削力
    previous_error = 200  # 误差
    integral = 0        # 积分项

    # 数据存储
    v_f_values = np.empty(int(T / dt), dtype=np.float64)     # 存储进给速度的变化
    F_values = np.empty(int(T / dt), dtype=np.float64)       # 存储切削力的变化

    # 仿真循环
    for i, _ in enumerate(time):
        # 计算切削力
        F_actual = F_actual + C * v_f * d + random.randint(-20, 200) / random.randint(100, 1000)  # 根据进给速度计算切削力

        # 计算误差
        error = F_target - F_actual

        # PID 控制器
        integral += error * dt
        derivative = (error - previous_error) / dt
        v_f = Kp * error + Ki * integral + Kd * derivative

        # 限制进给速度在合理范围内
        v_f = max(0, min(v_f, 200))  # 限制进给速度 (mm/min)

        # 更新前一个误差
        previous_error = error

        # 存储数据
        v_f_values[i] = v_f
        F_values[i] = F_actual
    
    return v_f_values, F_values

def calculate_lyapunov(data):
    perturbation_growth = np.array([
        np.log(abs((F_target - a) / F_target)) 
        for a in data
        ], dtype=np.float64)
    lyapunov_index = np.mean(perturbation_growth) / dt
    return lyapunov_index

SIMU_SIZE = 100
SIMU_RANGE = np.array([i*10e-7 for i in range(1, SIMU_SIZE+1)])


# 初始化结果存储数组
K = 10**4
result = np.empty((SIMU_SIZE, SIMU_SIZE, SIMU_SIZE), dtype=np.float64)


# 遍历 PID 增益空间
for x, p in enumerate(SIMU_RANGE):
    for y, i in enumerate(SIMU_RANGE):
        for z, d in enumerate(SIMU_RANGE):
            v_f_values, F_values = f(p, i, d)
            # 绘制图像
            # plt.figure(figsize=(12, 6))

            # # 绘制进给速度 v_f 随时间变化的图
            # plt.subplot(1, 2, 1)
            # plt.plot(time, v_f_values, color='blue', label='Feed rate (v_f)')
            # plt.title("Feed Rate vs Time")
            # plt.xlabel("Time (s)")
            # plt.ylabel("Feed Rate (mm/min)")
            # plt.grid(True)

            # # 绘制切削力 F_actual 随时间变化的图
            # plt.subplot(1, 2, 2)
            # plt.plot(time, F_values, color='red', label='Cutting Force (F_actual)')
            # plt.title("Cutting Force vs Time")
            # plt.xlabel("Time (s)")
            # plt.ylabel("Cutting Force (N)")
            # plt.grid(True)

            # # 显示图像
            # plt.tight_layout()
            # plt.show()


            lyapunov_index = calculate_lyapunov(F_values)
            result[x, y, z] = lyapunov_index
    print(f"No.\t{(x+1)*SIMU_SIZE**2}/{SIMU_SIZE**3} finished")

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

# 第1幅图：p vs i，颜色表示d维度的最优值
p, i = np.meshgrid(np.arange(1, SIMU_SIZE+1), np.arange(1, SIMU_SIZE+1))
d_best_1 = np.argmin(result, axis=2)  # 找到每个(p, i)组合对应的d的最优值
axes[0, 0].scatter(p.flatten(), i.flatten(), c=d_best_1.flatten(), cmap='viridis')
axes[0, 0].set_title("p vs i (best d as color)", fontsize=14)
axes[0, 0].set_xlabel("p", fontsize=12)
axes[0, 0].set_ylabel("i", fontsize=12)

# 第2幅图：p vs d，颜色表示i维度的最优值
i_best_2 = np.argmin(result, axis=1)  # 找到每个(p, d)组合对应的i的最优值
axes[0, 1].scatter(p.flatten(), i.flatten(), c=i_best_2.flatten(), cmap='viridis')
axes[0, 1].set_title("p vs d (best i as color)", fontsize=14)
axes[0, 1].set_xlabel("p", fontsize=12)
axes[0, 1].set_ylabel("d", fontsize=12)

# 第3幅图：i vs d，颜色表示p维度的最优值
p_best_3 = np.argmin(result, axis=0)  # 找到每个(i, d)组合对应的p的最优值
axes[1, 0].scatter(i.flatten(), p.flatten(), c=p_best_3.flatten(), cmap='viridis')
axes[1, 0].set_title("i vs d (best p as color)", fontsize=14)
axes[1, 0].set_xlabel("i", fontsize=12)
axes[1, 0].set_ylabel("d", fontsize=12)

# 第4幅图：柱状图，显示所有Lyapunov指数的分布
lyapunov_values = result.flatten()  # 拉平成一维数组
axes[1, 1].hist(lyapunov_values, bins=50, color='skyblue', edgecolor='black')
axes[1, 1].set_title("Lyapunov Index Distribution", fontsize=14)
axes[1, 1].set_xlabel("Lyapunov Index", fontsize=12)
axes[1, 1].set_ylabel("Frequency", fontsize=12)

# 调整布局
plt.tight_layout()
plt.show()