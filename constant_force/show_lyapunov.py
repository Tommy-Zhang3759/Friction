import torch
import matplotlib.pyplot as plt
import numpy as np

np_load = np.load("1e-06_9.999999999999999e-05_100*100.npy")
result = torch.from_numpy(np_load)

SIMU_SIZE = 100
SIMU_RANGE_P = torch.tensor([i * 1e-6 for i in range(1, SIMU_SIZE + 1)], device="cpu")
SIMU_RANGE_I = torch.tensor([i * 1e-6 for i in range(1, SIMU_SIZE + 1)], device="cpu")
SIMU_RANGE_D = torch.tensor([i * 1e-6 for i in range(1, SIMU_SIZE + 1)], device="cpu")

fig, axes = plt.subplots(2, 2, figsize=(15, 15))

p, i = torch.meshgrid(SIMU_RANGE_P, SIMU_RANGE_I)

# 第1幅图：p vs i，颜色表示d维度的最优值
d_best_1 = torch.argmin(result, axis=2)
axes[0, 0].scatter(p.flatten(), i.flatten(), c=d_best_1.flatten(), cmap='viridis')
axes[0, 0].set_title("p vs i (best d as color)", fontsize=14)
axes[0, 0].set_xlabel("p", fontsize=12)
axes[0, 0].set_ylabel("i", fontsize=12)

p, d = torch.meshgrid(SIMU_RANGE_P, SIMU_RANGE_D)

# 第2幅图：p vs d，颜色表示i维度的最优值
i_best_2 = torch.argmin(result, axis=1)
axes[0, 1].scatter(p.flatten(), d.flatten(), c=i_best_2.flatten(), cmap='viridis')
axes[0, 1].set_title("p vs d (best i as color)", fontsize=14)
axes[0, 1].set_xlabel("p", fontsize=12)
axes[0, 1].set_ylabel("d", fontsize=12)

i, d = torch.meshgrid(SIMU_RANGE_I, SIMU_RANGE_D)

# 第3幅图：i vs d，颜色表示p维度的最优值
p_best_3 = torch.argmin(result, axis=0)
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