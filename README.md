# deblur_gan_ros

ROS 2 node wrapping Ghost-DeblurGAN for real-time image deblurring on live topics. It subscribes to a `sensor_msgs/Image` stream, restores sharp frames using a lightweight GAN, and republishes the result.

- Package: `deblur_gan_ros` (ament_python)
- Default model: GhostNet + HIN + Ghost modules (included DeblurGAN model too (see https://github.com/emanuelenencioni/Ghost-DeblurGAN))
- Pretrained weights included
- GPU acceleration optional (PyTorch CUDA), CPU fallback available


## Requirements

- OS: Ubuntu 22.04 (recommended) or compatible Linux
- ROS 2 Humble or newer (rclpy, sensor_msgs, cv_bridge)
- Python 3.10+ (tested with 3.10 conda env)
- Optional GPU: NVIDIA driver on host; NVIDIA Container Toolkit if running in Docker


## Install (native)

Assuming a standard workspace at `~/ros2_ws` and this repo cloned at `~/ros2_ws/src/deblur_gan_ros`.

1) Install ROS 2 system deps for this package

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions
cd ~/ros2_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

if you want to use a virtual env, install also `rospkg` and `colcon-common-extensions` into it.

2) Install Python dependencies (PyTorch + Ghost-DeblurGAN requirements)
Follow the installation instructions from the [Ghost-DeblurGAN repository](https://github.com/emanuelenencioni/Ghost-DeblurGAN).


3) Build and source

**Note**: modify `setup.cfg` in your package:
```ini
[options]
executable = /path/to/your/virtual_env_or_system_env/bin/python3
```

```bash
cd ~/ros2_ws
colcon build --packages-select deblur_gan_ros
source install/setup.bash
```

## Run

Default launch (subscribes to `image_raw`, publishes to `deblur_gan/image_raw`):

```bash
ros2 launch deblur_gan_ros deblur_gan.py
```

ZED example (right camera topic):

```bash
ros2 launch deblur_gan_ros gdgan_zed.py
```

Override parameters on the command line:

```bash
ros2 launch deblur_gan_ros gdgan_launch.py \
	input_topic:=/camera/image_color \
	output_topic:=/deblur_gan/image \
	weights_name:=fpn_ghostnet_gm_hin.h5 \
	model_name:=fpn_ghostnet \
	cuda:=true
```

Node parameters:

- `input_topic` (string, default: `image_raw`) – subscribed image topic (encoding BGR8 handled via cv_bridge)
- `output_topic` (string, default: `deblur_gan/image_raw`) – published deblurred image topic (BGR8)
- `weights_name` (string, default: `fpn_ghostnet_gm_hin.h5`) – filename from the packaged weights
- `model_name` (string, default: `fpn_ghostnet`) – model architecture
- `cuda` (bool, default: `true`) – enable GPU (PyTorch CUDA); set to `false` for CPU

Packaged weights and config are installed to:

- `share/deblur_gan_ros/weights/` (e.g., `fpn_ghostnet_gm_hin.h5`, `fpn_mobilenet_v2.h5`)
- `share/deblur_gan_ros/config/config.yaml`


## Topics

- Subscribes: `sensor_msgs/msg/Image` on `input_topic`
- Publishes: `sensor_msgs/msg/Image` (encoding: BGR8) on `output_topic`


## Docker

Use your external Dockerfiles from: https://github.com/emanuelenencioni/docker_images

This lets you build a ROS 2 + (optional) CUDA + PyTorch image once and reuse it for this package. Below are generic steps; consult that repo’s README for the exact Dockerfile path you prefer (GPU or CPU).

### 1) Prerequisites

- Docker Engine 24+
- For GPU runs: NVIDIA drivers on host and NVIDIA Container Toolkit

Install NVIDIA Container Toolkit (Ubuntu/Debian):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
	sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
	sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU visibility:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```

### 2) Build the base image from docker_images

I have an already tested Docker image that I use for research. You can use that.

Follow the instructions on https://github.com/emanuelenencioni/docker_images on how to build the image and run it.

Tips:

- Add more `-v` mounts for datasets or bags as needed (read-only when possible).
- If your base image already contains a built workspace, you can skip the `colcon build` step and only `source` + `ros2 launch`.


## Notes and tips

- Ensure your incoming images have an encoding convertible by `cv_bridge` to `bgr8`.
- For CPU-only, set `cuda:=false` to avoid unnecessary CUDA context creation.
- The node pads images to multiples of 32 internally and crops back to original size.
- If you see out-of-memory on GPU, reduce input resolution or switch to CPU.


## Acknowledgements

This package integrates the Ghost-DeblurGAN implementation (IROS 2022) adapted from DeblurGANv2. See `ghost_deblurgan/README.md` for details.

## License

- Node code: Apache-2.0 (see `package.xml`)
- Ghost-DeblurGAN components: see `ghost_deblurgan/LICENSE`

