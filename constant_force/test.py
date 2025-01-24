import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# 仿真参数
T = 1             # 仿真时间 (秒)
dt = 0.001        # 时间步长 (秒)
time = torch.arange(0, T, dt)

# 参数设定
C = 1.2             # 材料常数
d = 10              # 钻头直径 (mm)
F_target = 200      # 目标切削力 (N)

SIMU_SIZE = 20  # 降低模拟尺寸以便实时效果
SIMU_RANGE_P = torch.tensor([i * 1e-6 for i in range(1, SIMU_SIZE + 1)])
SIMU_RANGE_I = torch.tensor([i * 1e-6 for i in range(1, SIMU_SIZE + 1)])
SIMU_RANGE_D = torch.tensor([i * 1e-6 for i in range(1, SIMU_SIZE + 1)])

# 结果存储
result = torch.empty((SIMU_SIZE, SIMU_SIZE, SIMU_SIZE))
status = np.zeros((SIMU_SIZE, SIMU_SIZE, SIMU_SIZE))  # 0: 未完成, 1: 完成

def f(Kp, Ki, Kd):
    v_f = torch.zeros_like(time)
    F_actual = torch.zeros_like(time)
    previous_error = F_target
    integral = torch.zeros_like(time)

    for i in range(len(time)):
        random_perturbation = torch.randint(-200, 2000, (1,), dtype=torch.float32) / torch.randint(100, 1000, (1,), dtype=torch.float32)
        F_actual[i] = F_actual[i - 1] + C * v_f[i - 1] * d + random_perturbation

        error = F_target - F_actual[i]
        integral[i] = integral[i - 1] + error * dt
        derivative = (error - previous_error) / dt
        v_f[i] = Kp * error + Ki * integral[i] + Kd * derivative
        v_f[i] = torch.clamp(v_f[i], 0, 200)

        previous_error = error

    return v_f, F_actual

def calculate_lyapunov(data):
    perturbation_growth = torch.log(torch.abs((F_target - data) / F_target))
    lyapunov_index = torch.mean(perturbation_growth) / dt
    return lyapunov_index

def simulate_pid(index):
    x, y, z = index
    p, i, d = SIMU_RANGE_D[x], SIMU_RANGE_I[y], SIMU_RANGE_P[z]
    _, F_values = f(p, i, d)
    lyapunov_index = calculate_lyapunov(F_values)
    return x, y, z, lyapunov_index

# 初始化绘图
fig, ax = plt.subplots(figsize=(10, 10))
p, i, d = np.meshgrid(range(SIMU_SIZE), range(SIMU_SIZE), range(SIMU_SIZE))
sc = ax.scatter(p.flatten(), i.flatten(), c=status.flatten(), cmap="RdYlGn", s=100)
ax.set_title("Simulation Progress")
ax.set_xlabel("P Index")
ax.set_ylabel("I Index")

# 调度模拟任务
def run_simulations():
    indices = [(x, y, z) for x in range(SIMU_SIZE) for y in range(SIMU_SIZE) for z in range(SIMU_SIZE)]
    with ThreadPoolExecutor() as executor:
        for x, y, z, lyapunov_index in executor.map(simulate_pid, indices):
            result[x, y, z] = lyapunov_index
            status[x, y, z] = 1  # 设置为完成

# 更新可视化
def update(frame):
    sc.set_array(status.flatten())
    return sc,

# 异步运行模拟任务
executor = ThreadPoolExecutor()
executor.submit(run_simulations)

# 实时更新图像
ani = FuncAnimation(fig, update, frames=100, interval=500, repeat=False)

plt.show()
