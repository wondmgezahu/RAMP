# RAMP
Real-Time Adaptive Motion Planning via Point Cloud-Guided, Energy-Based Diffusion and Potential Fields

## Method Overview
Our experiments demonstrate how real-time adaptive motion planning can be effectively applied to pursuit-evasion scenarios with non-holonomic constraints. The RAMP framework leverages point cloud data and energy-based diffusion techniques to generate efficient trajectories in the presence of dynamic agent and static obstacles.
## Results

Below are simulation and physical robot demonstrations showcasing the capabilities of our RAMP framework.

### Simulation Results


<div style="display: flex; justify-content: space-between; align-items: center;">
  <img src="https://github.com/user-attachments/assets/d8121aaa-9153-4585-8ca5-7d65280ffa0f" height="350" width="350" alt="Maze2D compositional results" style="margin-right: 10%;">
  <img src="https://github.com/user-attachments/assets/7320c893-1c45-41c0-969c-c7e76c36ea07" height="350" width="400" alt="Maze3D navigation with 25 obstacles">
</div>


*Left: Maze2D compositional results shows generalization capabilities. Right: Static obstacle avoidance in Maze3D environment with 20+ obstacles.*

# Physical Robot Demonstrations

## Experimental Setup

Our pursuit-evasion scenario experiments were conducted using the [QCar1 by Quanser](https://www.quanser.com/products/qcar/), a 1/10-scale autonomous vehicle platform designed for robotics and AI research. The platform features onboard sensors including an IMU, RGB-D camera, and LiDAR, along with motion capture markers for external tracking system integration. This configuration makes the QCar1 ideal for research in localization, control, and navigation applications.

## Large Environment 

### Workspace Configuration
All experiments are conducted under the following conditions:
- **Environment**: 6×6m² experimental area
- **Robots**: Two autonomous vehicles - blue evader and red pursuer  
- **Objective**: Evader must reach designated goal while avoiding capture
- **Constraints**: All robots operate under non-holonomic motion constraints
- **Visualization**: Split-screen videos showing lab setup (bottom) and trajectory visualization (top)


## Experimental Scenarios

Four pursuit-evasion experiments evaluated under varying obstacle densities:

**Experimental Conditions:**
- **Scenario 1**: 4 static obstacles (baseline environment)
- **Scenario 2**: 6 static obstacles (additional 2 unseen obstacles)

## Experimental Videos

<video src="https://github.com/user-attachments/assets/caa43491-58ce-4817-9da9-8a3b4bb78d5b" controls="controls" style="max-width: 100%;">
</video>


**📥 [Download Full Quality](https://github.com/user-attachments/assets/d010fd87-10b7-4593-865e-b15ee8a3fa59)** *(32.9 MB, 1:10 duration)*  

*Demonstrates all four pursuit-evasion scenarios with real robot validation.*

*Note: Download required for optimal viewing due to video format.*

## Small Environment 

### Workspace Configuration
All experiments are conducted under the following conditions:
- **Environment**: 2×2m² experimental area
- **Robots**: Two autonomous vehicles - green evader and red pursuer  
- **Objective**: Evader must reach designated goal while avoiding capture
- **Constraints**: All robots operate under non-holonomic motion constraints
- **Visualization**: Split-screen videos showing lab setup (right) and trajectory visualization (left)

## Experimental Scenarios

Three pursuit-evasion experiments demonstrate our method performance under open and static obstacle configurations.

## Experimental Videos

https://github.com/user-attachments/assets/7943ebcd-91ad-4839-9b45-e628b23893c8

