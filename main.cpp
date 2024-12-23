#include <iostream>
#include "main.hpp"

using namespace std;

Main_motor::Main_motor(uint32_t pwmChannel) : pwmChannel(pwmChannel), current_pos(0), current_v(0),
                                              target_pos(0), target_v(0), initPosition(0), acceleration(0) {}

Main_motor::~Main_motor() {}

bool Main_motor::setSpeed(mps s) {
    current_v = s;
    // 更新PWM占空比
    // 示例代码：调用HAL库调整PWM信号
    __HAL_TIM_SET_COMPARE(&htim1, pwmChannel, static_cast<uint32_t>(s * 1000));
    return true;
}

bool Main_motor::setTargetPosition(meter x) {
    target_pos = x;
    return true;
}

meter Main_motor::getPosition() const {
    return current_pos;
}

mps Main_motor::getSpeed() const {
    return current_v;
}

// --- PressureControl 实现 ---
PressureControl::PressureControl(double kP, double kI, double kD)
    : kP(kP), kI(kI), kD(kD), integral(0), previousError(0), targetPressure(0), currentPressure(0) {}

void PressureControl::setTargetPressure(double pressure) {
    targetPressure = pressure;
}

void PressureControl::updatePressure(double pressure) {
    currentPressure = pressure;
}

double PressureControl::calculateAdjustment() {
    double error = targetPressure - currentPressure;
    integral += error;
    double derivative = error - previousError;
    previousError = error;

    return kP * error + kI * integral + kD * derivative;
}

// --- Task 类实现 ---
Task::Task(Main_motor feed, Main_motor drill, PressureControl control)
    : feedMotor(feed), drillMotor(drill), pressureControl(control) {}

int Task::run() {
    // 更新压力传感器数据
    double currentPressure = readPressure(); // 读取压力传感器数据
    pressureControl.updatePressure(currentPressure);

    // 计算进钻速度调整
    double adjustment = pressureControl.calculateAdjustment();
    feedMotor.setSpeed(feedMotor.getSpeed() + adjustment);

    // 运行电机
    feedMotor.run();
    drillMotor.run();

    return 0;
}
