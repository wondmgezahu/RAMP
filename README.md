# RAMP
Real-Time Adaptive Motion Planning via Point Cloud-Guided, Energy-Based Diffusion and Potential Fields

This documentation presents physical robot demonstrations for pursuit-evasion scenarios implemented using our RAMP framework.
### Details of RC QCar
## New Physical Robot Experiments

### Experiment Workspace Setup
All experiments are conducted under the following conditions:

- 6×6m² experimental environment
- Two autonomous robots: a green evader and a red pursuer
- The evader must reach a designated goal while avoiding capture
- All robots operate under non-holonomic motion constraints
- Videos show split-screen views: lab setup (right) and trajectory visualization (left)
- Purple dashed line indicates the evader's initially planned high-level trajectory
## Previous Physical Robot Experiments 

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

### Experiment 2: With Static Obstacle

https://github.com/user-attachments/assets/052c0402-d533-4a82-b35e-da21c329971d

The second experiment introduces an unseen static obstacle at the center of the lab. This setup uses a new set of initial positions and a different final goal for the evader. The video clearly shows the purple area indicating the static obstacle's position.

### Experiment 3: Comparative Study

https://github.com/user-attachments/assets/c2c5c8a1-562a-47ed-8c25-1bd3ccf357a6

The third experiment replicates Experiment 2's initial conditions and goal position, but removes the center obstacle to highlight its effect on robot behavior. By maintaining identical starting conditions to Experiment 2 but eliminating the obstacle, this experiment creates a direct comparison that isolates the impact of the environmental obstacle on pursuit-evasion dynamics.

## Implementation Details

Our experiments demonstrate how real-time adaptive motion planning can be effectively applied to pursuit-evasion scenarios with non-holonomic constraints. The RAMP framework leverages point cloud data and energy-based diffusion techniques to generate efficient trajectories in the presence of dynamic agents and static obstacles.
