# Multi-Model Perception System in ROS 2

A hybrid object detection + classification system combining **YOLOv8 (for detection)** and **ResNet50 (for refined classification)**, integrated into the **ROS 2 Humble** environment.

## 🧩 Packages Included

- `bcr_bot`: Robot simulation package **cloned** from [blackcoffeerobotics/bcr_bot](https://github.com/blackcoffeerobotics/bcr_bot)
- `resnet50-yolov8-template`: Custom multi-model perception system:
  - `yolov8_template_msgs`: Custom message definitions
  - `template_recognition`: ROS nodes running YOLOv8 + ResNet50 models

## 🛠️ Setup Instructions

```bash
# Clone this repo
git clone https://github.com/hayetem/multi-model-implementation-ros2.git
cd multi-model-implementation-ros2

# Install Python dependencies
pip install numpy==1.24.4 tensorflow torch ultralytics cv_bridge

# Build the workspace
colcon build --packages-select yolov8_template_msgs template_recognition
source install/setup.bash

# Launch the multi-model perception node
ros2 launch template_recognition launch_yolov8.launch.py



NOTE: the project must contain a yolov8.pt and a resnet50.keras files 
