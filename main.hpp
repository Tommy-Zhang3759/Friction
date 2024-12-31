#include <cstdint>
#include <cmath>

#ifndef MOTOR_CONTROL
#define MOTOR_CONTROL
#endif

using namespace std;

typedef double meter;   // m
typedef double mps;     // m/s
typedef double mps2;    // m/s^2

class Main_motor {
private:
    uint32_t pwmChannel;   // PWM通道，用于控制电机速度
    meter current_pos;
    mps current_v;
    meter target_pos;
    mps target_v;
    meter initPosition;
    mps2 acceleration;

    void runner();         // 运行任务，控制电机

public:
    Main_motor(uint32_t pwmChannel);
    ~Main_motor();

    void run();
    bool setSpeed(mps s);
    bool setAcceleration(mps2 a);
    bool setTargetPosition(meter x);
    bool setInitPosition(meter x);

    meter getPosition() const;
    mps getSpeed() const;
};

class PressureControl {
private:
    double zeroPoint;
    double targetPressure;  // 目标压力
    double currentPressure; // 当前压力
    double kP, kI, kD;      // PID控制参数
    double integral;        // 积分项
    double previousError;   // 上一次误差

public:
    PressureControl(double kP, double kI, double kD);

    void setTargetPressure(double pressure);
    double getTargetPressure() const;
    void _updatePressure(double pressure);
    double calculateAdjustment();
};

class Task {
private:
    Main_motor feedMotor;   // 进钻电机
    Main_motor drillMotor;  // 钻头驱动电机
    PressureControl pressureControl; // 压力控制器

public:
    Task(Main_motor feed, Main_motor drill, PressureControl control);

    int run();
};
