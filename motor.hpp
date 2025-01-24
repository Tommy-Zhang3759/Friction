#ifndef MOTOR_H
#define MOTOR_H

#include <cstdint>
#include <cmath>

using namespace std;

typedef double meter;   // m
typedef double mps;     // m/s
typedef double mps2;    // m/s^2

typedef int step;
typedef double rad;

static uint32_t clock_frequency = 64000000;

class Motor {
private:
    step _position;
    rad _step_len;

    uint32_t aux_timer;
    uint32_t pwm_timer;

    uint32_t pwm_pin;
    uint32_t pwm_io_grup;
public:
    Motor(uint32_t pwm_timer, uint32_t aux_timer, uint32_t pwm_pin, uint32_t pwm_io_grup, rad step_len);

    ~Motor();

    void run(uint32_t step_num);
    void zero();

    step position();
    rad step_len();
};


#endif