import numpy as np
import matplotlib.pyplot as plt
import random
from concurrent.futures import ThreadPoolExecutor
import time as timer

begin_time = timer.time()

T = 1             # 仿真时间 (秒)
dt = 0.001       # 时间步长 (秒)
time = np.arange(0, T, dt)

# 参数设定
C = 1.2             # 材料常数
d = 10              # 钻头直径 (mm)
F_target = 200      # 目标切削力 (N)

SIMU_SIZE = 30
SIMU_RANGE_P = np.array([i * 1e-6 for i in range(1, SIMU_SIZE + 1)])
SIMU_RANGE_I = np.array([i * 1e-6 for i in range(1, SIMU_SIZE + 1)])
SIMU_RANGE_D = np.array([i * 1e-6 for i in range(1, SIMU_SIZE + 1)])

# 初始结果存储数组
result = np.empty((SIMU_SIZE, SIMU_SIZE, SIMU_SIZE), dtype=np.float64)
simu_data = np.empty((SIMU_SIZE, SIMU_SIZE, SIMU_SIZE, 2), dtype=list)

# 定义仿真函数
def f(Kp, Ki, Kd):
    v_f = 0             # 初始进给速度
    F_actual = 0        # 初始切削力
    previous_error = 200  # 初始误差
    integral = 0        # 积分项

    v_f_values = np.empty(int(T / dt), dtype=np.float64)
    F_values = np.empty(int(T / dt), dtype=np.float64)

    for i, _ in enumerate(time):
        F_actual += C * v_f * d + random.randint(-2, 20) / random.randint(10, 100) * v_f
        error = F_target - F_actual

        integral += error * dt
        derivative = (error - previous_error) / dt
        v_f = Kp * error + Ki * integral + Kd * derivative

        v_f = max(0, min(v_f, 200))
        previous_error = error

        v_f_values[i] = v_f
        F_values[i] = F_actual

    return v_f_values, F_values

def calculate_lyapunov(data):
    perturbation_growth = np.array([
        np.log(abs((F_target - a) / F_target)) for a in data
    ], dtype=np.float64)

    p = 0
    for i in range(perturbation_growth):
        perturbation_growth[i] /= dt**(i + 1)
        if perturbation_growth[i] <= 0:
            p = i
    return perturbation_growth, p
            

def simulate_pid(index):
    x, y, z = index
    p, i, d = SIMU_RANGE_P[x], SIMU_RANGE_I[y], SIMU_RANGE_D[z]
    v_f_values, F_values = f(p, i, d)
    
    simu_data[x, y, z, 0] = v_f_values
    simu_data[x, y, z, 1] = F_values
    lyapunov_index = calculate_lyapunov(F_values)
    # print(f"x:{x}/{SIMU_SIZE} y:{y}/{SIMU_SIZE} z:{z}/{SIMU_SIZE}")
    return x, y, z, lyapunov_index

# 多线程计算
def main():
    indices = [(x, y, z) for x in range(SIMU_SIZE) for y in range(SIMU_SIZE) for z in range(SIMU_SIZE)]

    with ThreadPoolExecutor() as executor:
        for x, y, z, lyapunov_index in executor.map(simulate_pid, indices):
            result[x, y, z] = lyapunov_index

    print("Simulation finished.")

    np.savez(f"{SIMU_RANGE_D[0]}_{SIMU_RANGE_D[-1]}_{SIMU_RANGE_I[0]}_{SIMU_RANGE_I[-1]}_{SIMU_RANGE_P[0]}_{SIMU_RANGE_P[-1]}_{SIMU_SIZE}x{SIMU_SIZE}_{begin_time}.npz",
              lyapunov_index=result,
              simu_data=simu_data
              )

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    p, i = np.meshgrid(SIMU_RANGE_P, SIMU_RANGE_I)

    # 第1幅图：p vs i，颜色表示d维度的最优值
    d_best_1 = np.argmin(result, axis=2)
    axes[0, 0].scatter(p.flatten(), i.flatten(), c=d_best_1.flatten(), cmap='viridis')
    axes[0, 0].set_title("p vs i (best d as color)", fontsize=14)
    axes[0, 0].set_xlabel("p", fontsize=12)
    axes[0, 0].set_ylabel("i", fontsize=12)

    p, d = np.meshgrid(SIMU_RANGE_P, SIMU_RANGE_D)

    # 第2幅图：p vs d，颜色表示i维度的最优值
    i_best_2 = np.argmin(result, axis=1)
    axes[0, 1].scatter(p.flatten(), d.flatten(), c=i_best_2.flatten(), cmap='viridis')
    axes[0, 1].set_title("p vs d (best i as color)", fontsize=14)
    axes[0, 1].set_xlabel("p", fontsize=12)
    axes[0, 1].set_ylabel("d", fontsize=12)

    i, d = np.meshgrid(SIMU_RANGE_I, SIMU_RANGE_D)

    # 第3幅图：i vs d，颜色表示p维度的最优值
    p_best_3 = np.argmin(result, axis=0)
    axes[1, 0].scatter(i.flatten(), d.flatten(), c=p_best_3.flatten(), cmap='viridis')
    axes[1, 0].set_title("i vs d (best p as color)", fontsize=14)
    axes[1, 0].set_xlabel("i", fontsize=12)
    axes[1, 0].set_ylabel("d", fontsize=12)

    # 第4幅图：柱状图，显示所有Lyapunov指数的分布
    lyapunov_values = result.flatten()
    axes[1, 1].hist(lyapunov_values, bins=50, color='skyblue', edgecolor='black')
    axes[1, 1].set_title("Lyapunov Index Distribution", fontsize=14)
    axes[1, 1].set_xlabel("Lyapunov Index", fontsize=12)
    axes[1, 1].set_ylabel("Frequency", fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    print(f"spent: {timer.time()-begin_time}")
