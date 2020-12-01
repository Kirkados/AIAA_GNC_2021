# AIAA_GNC_2021

A chaser quadrotor is tasked with learning how to track a target quadrotor on its own using deep reinforcement learning. In order to have the technique transfer well from training in simulation to a real experimental facility, the deep reinforcement learning is restricted to guidance-only. Velocity-based and acceleration-based guidance approaches are compared in the paper:
Hovell, K., Ulrich, S., and Bronz, M., “Acceleration-based Quadrotor Guidance Under Time Delays Using Deep Reinforcement Learning,” AIAA Guidance, Navigation, and Control Conference, Nashville, TN, 11-21 Jan, 2021.

This GitHub respository contains all the code used in the paper. It is a D4PG implementation of both a velocity- and acceleration-based guidance system for quadrotors. In the paper, we show that the acceleration-based implementation is more approriate for second-order systems.

To run, first modify settings in environment_quad1_accel.py and settings.py. Then run python3 on the main.py to begin training.

Tensorflow version 1.15 was used for this work.
