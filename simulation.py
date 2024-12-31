import numpy as np
import matplotlib.pyplot as plt

# 参数设定
C = 1.2             # 材料常数
d = 10              # 钻头直径 (mm)
F_target = 200      # 目标切削力 (N)
# v_f_target = 100    # 目标进给速度 (mm/min)

# PID 控制器的增益
Kp = 0.5
Ki = 0.1
Kd = 0.05

# 仿真时间
T = 100             # 仿真时间 (秒)
dt = 0.1            # 时间步长 (秒)
time = np.arange(0, T, dt)

# 初始条件
v_f = 0             # 初始进给速度
F_actual = 0        # 初始切削力
previous_error = 0  # 误差
integral = 0        # 积分项

# 数据存储
v_f_values = []     # 存储进给速度的变化
F_values = []       # 存储切削力的变化

# 仿真循环
for t in time:
    # 计算切削力
    F_actual = C * v_f * d  # 根据进给速度计算切削力

    # 计算误差
    error = F_target - F_actual

    # PID 控制器
    integral += error * dt
    derivative = (error - previous_error) / dt
    v_f = Kp * error + Ki * integral + Kd * derivative

    # 限制进给速度在合理范围内
    v_f = max(0, min(v_f, 20000))  # 限制进给速度 (mm/min)

    # 更新前一个误差
    previous_error = error

    # 存储数据
    v_f_values.append(v_f)
    F_values.append(F_actual)

# 可视化结果
plt.figure(figsize=(12, 6))

# 进给速度 vs 时间
plt.subplot(2, 1, 1)
plt.plot(time, v_f_values, 'b', label='Feed Speed (mm/min)')
plt.xlabel('Time (s)')
plt.ylabel('Feed Speed (mm/min)')
plt.title('Feed Speed vs Time')
plt.grid(True)

# 切削力 vs 时间
plt.subplot(2, 1, 2)
plt.plot(time, F_values, 'r', label='Cutting Force (N)')
plt.xlabel('Time (s)')
plt.ylabel('Cutting Force (N)')
plt.title('Cutting Force vs Time')
plt.grid(True)

plt.tight_layout()
plt.show()
