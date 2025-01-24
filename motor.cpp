#include <iostream>
#include "motor.hpp"

using namespace std;

Motor::Motor(uint32_t pwm_timer, uint32_t aux_timer, uint32_t pwm_pin, uint32_t pwm_io_grup, rad step_len)
    : pwm_timer(pwm_timer), aux_timer(aux_timer), pwm_pin(pwm_pin), pwm_io_grup(pwm_io_grup), _step_len(step_len) {
    this->_position = 0;
}

Motor::~Motor()
{
}

void Motor::run(uint32_t step_num){
    static uint32_t pwm_timer_period = 1000000;
    static uint32_t minor_timer_period = pwm_timer_period * step_num / clock_frequency;

    return;
}

void Motor::zero(){
    this->_position = 0;
    return;
}

step Motor::position(){
    return this->_position;
}

rad Motor::step_len(){
    return this->_step_len;
}