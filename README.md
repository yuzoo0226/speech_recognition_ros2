# speech recognition pkgs

## How to Clone

```bash
git clone --recursive https://github.com/yuzoo0226/speech_recognition_ros2.git
```

## Environments

- Ubuntu22.04
- Humble

## 音声トピック配信ノードのみのビルドと起動

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to audio_common
source install/setup.bash
ros2 run audio_common audio_captuer_node
```


## 認識ノードの起動

```bash
ros2 run speech_recognition_node speech_recognition_node
```
