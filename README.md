# RAMP
Real-Time Adaptive Motion Planning via Point Cloud-Guided, Energy-Based Diffusion and Potential Fields
## Physical Robot Experiments
This repository demonstrates physical robot experiments for pursuit-evasion scenarios using our RAMP framework.

All experiments are conducted under the following conditions:

- 2×2m² experimental environment
- Two autonomous robots: a green evader and a red pursuer
- The evader must reach a designated goal while avoiding capture
- All robots operate under non-holonomic motion constraints
- Videos show split-screen views: lab setup (right) and trajectory visualization (left)
- Purple dashed line indicates the evader's initially planned high-level trajectory

### Experiment 1: Open Environment

https://github.com/user-attachments/assets/100e4e85-387d-4aba-9923-ea0c5fc3bc4c

This experiment presents a pursuit-evasion scenario without static obstacles. Both robots are initially positioned at the top of the field, with the evader attempting to reach its designated goal on the left-hand side. The open environment provides sufficient freedom of movement while accounting for the limitations of our lab setup and the non-holonomic nature of the robots.

