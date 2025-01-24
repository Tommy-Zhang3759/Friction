import torch
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time as timer

begin_time = timer.time()

# 仿真参数
T = 1             # 仿真时间 (秒)
dt = 0.001       # 时间步长 (秒)
time = torch.arange(0, T, dt, device="mps")

# 参数设定
C = 1.2             # 材料常数
d = 10              # 钻头直径 (mm)
F_target = 200      # 目标切削力 (N)

SIMU_SIZE = 10
SIMU_RANGE_P = torch.tensor([i * 1e-6 for i in range(1, SIMU_SIZE + 1)], device="mps")
SIMU_RANGE_I = torch.tensor([i * 1e-6 for i in range(1, SIMU_SIZE + 1)], device="mps")
SIMU_RANGE_D = torch.tensor([i * 2e-7 for i in range(1, SIMU_SIZE + 1)], device="mps")
# 结果存储
result = torch.empty((SIMU_SIZE, SIMU_SIZE, SIMU_SIZE), device="mps")

def f(Kp, Ki, Kd):
    v_f = torch.zeros_like(time, device="mps")
    F_actual = torch.zeros_like(time, device="mps")
    previous_error = torch.tensor(F_target, device="mps")
    integral = torch.zeros_like(time, device="mps")

    for i in range(len(time)):
        random_perturbation = torch.randint(-200, 2000, (1,), device="mps", dtype=torch.float32) / torch.randint(100, 1000, (1,), device="mps", dtype=torch.float32)
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

def main():
    indices = [(x, y, z) for x in range(SIMU_SIZE) for y in range(SIMU_SIZE) for z in range(SIMU_SIZE)]

    with ThreadPoolExecutor() as executor:
        for x, y, z, lyapunov_index in executor.map(simulate_pid, indices):
            result[x, y, z] = lyapunov_index

    print("Simulation finished.")

    # 保存结果
    torch.save(result.cpu(), f"{SIMU_RANGE_D[0]}_{SIMU_RANGE_D[-1]}_{SIMU_RANGE_I[0]}_{SIMU_RANGE_I[-1]}_{SIMU_RANGE_P[0]}_{SIMU_RANGE_P[-1]}_{SIMU_SIZE}x{SIMU_SIZE}_{begin_time}.pt")

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    p, i = torch.meshgrid(SIMU_RANGE_P, SIMU_RANGE_I).cpu()

    # 第1幅图：p vs i，颜色表示d维度的最优值
    d_best_1 = torch.argmin(result, axis=2).cpu()
    axes[0, 0].scatter(p.flatten(), i.flatten(), c=d_best_1.flatten(), cmap='viridis')
    axes[0, 0].set_title("p vs i (best d as color)", fontsize=14)
    axes[0, 0].set_xlabel("p", fontsize=12)
    axes[0, 0].set_ylabel("i", fontsize=12)

    p, d = torch.meshgrid(SIMU_RANGE_P, SIMU_RANGE_D).cpu()

    # 第2幅图：p vs d，颜色表示i维度的最优值
    i_best_2 = torch.argmin(result, axis=1).cpu()
    axes[0, 1].scatter(p.flatten(), d.flatten(), c=i_best_2.flatten(), cmap='viridis')
    axes[0, 1].set_title("p vs d (best i as color)", fontsize=14)
    axes[0, 1].set_xlabel("p", fontsize=12)
    axes[0, 1].set_ylabel("d", fontsize=12)

    i, d = torch.meshgrid(SIMU_RANGE_I, SIMU_RANGE_D).cpu()

    # 第3幅图：i vs d，颜色表示p维度的最优值
    p_best_3 = torch.argmin(result, axis=0).cpu()
    axes[1, 0].scatter(i.flatten(), d.flatten(), c=p_best_3.flatten(), cmap='viridis')
    axes[1, 0].set_title("i vs d (best p as color)", fontsize=14)
    axes[1, 0].set_xlabel("i", fontsize=12)
    axes[1, 0].set_ylabel("d", fontsize=12)

    # 第4幅图：柱状图，显示所有Lyapunov指数的分布
    lyapunov_values = result.flatten().cpu()
    axes[1, 1].hist(lyapunov_values, bins=50, color='skyblue', edgecolor='black')
    axes[1, 1].set_title("Lyapunov Index Distribution", fontsize=14)
    axes[1, 1].set_xlabel("Lyapunov Index", fontsize=12)
    axes[1, 1].set_ylabel("Frequency", fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    print(f"spent: {timer.time()-begin_time}")
