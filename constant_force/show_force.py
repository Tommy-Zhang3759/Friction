import matplotlib.pyplot as plt
import numpy as np

T = 1             # 仿真时间 (秒)
dt = 0.001       # 时间步长 (秒)
time = np.arange(0, T, dt)

data_file = np.load("2e-07_6e-06_1e-06_2.9999999999999997e-05_1e-06_2.9999999999999997e-05_30x30_1735740583.079318.npz",
                    allow_pickle=True
                    )

plt.figure(figsize=(12, 6))

simu_data = data_file["simu_data"]

v_f_values = simu_data[:, :, :, 0]
F_values = simu_data[:, :, :, 1]

X_SELECT = 5
Y_SELECT = 10
Z_SELECT = 10
def display_force(x, y, z):
    # 绘制进给速度 v_f 随时间变化的图
    plt.subplot(1, 2, 1)
    plt.plot(time, v_f_values[x, y, z], color='blue', label='Feed rate (v_f)')
    plt.title("Feed Rate vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Feed Rate (mm/min)")
    plt.grid(True)

    # 绘制切削力 F_actual 随时间变化的图
    plt.subplot(1, 2, 2)
    plt.plot(time, F_values[x, y, z], color='red', label='Cutting Force (F_actual)')
    plt.title("Cutting Force vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Cutting Force (N)")
    plt.grid(True)

    # 显示图像
    plt.tight_layout()
    plt.show()

for i in range(30):
    for j in range(30):
        for k in range(30):
            display_force(i, j, k)