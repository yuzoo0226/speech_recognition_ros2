# speech recognition pkgs

## How to Clone

```bash
git clone --recursive https://github.com/yuzoo0226/speech_recognition_ros2.git
```

## Environments

- Ubuntu22.04
- Humble

## How to launch audio publisher/capture node

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to audio_common
source install/setup.bash
ros2 run audio_common audio_captuer_node
```


## How to launch Speech recognition server node

```bash
ros2 run speech_recognition_node speech_recognition_node
```
