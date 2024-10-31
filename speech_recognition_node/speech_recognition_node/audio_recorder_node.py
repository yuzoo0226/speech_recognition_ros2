#!/usr/bin/env python3

import os
import wave
import struct
import numpy as np
from typing import List
# from silero_vad import VADIterator, load_silero_vad

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool

from audio_common.utils import msg_to_array
from audio_common_msgs.msg import AudioStamped
from ament_index_python.packages import get_package_share_directory


class SileroVadNode(Node):

    def __init__(self) -> None:

        super().__init__("audio_recorder_node")

        self.recording = True
        self.data: List[int] = []

        self.declare_parameter("enabled", True)
        self.enabled = self.chunk = (self.get_parameter("enabled").get_parameter_value().bool_value)
        self.declare_parameter("threshold", 0.5)
        self.threshold = self.chunk = (self.get_parameter("threshold").get_parameter_value().double_value)
        self.record_duration = 5  # sec

        self.output_dir = os.path.join(get_package_share_directory('speech_recognition_node'), 'io/output')
        self._sub = self.create_subscription(AudioStamped, "audio", self.audio_cb, qos_profile_sensor_data)

        self.get_logger().info("audio recorder node started")

    @staticmethod
    def int2float(audio_data: np.ndarray) -> np.ndarray:

        if audio_data is None or len(audio_data) == 0:
            return None

        from_type = type(audio_data[0])

        if from_type == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128) / 128.0

        elif from_type == np.int8:
            audio_data = audio_data.astype(np.float32) / 128.0

        elif from_type == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        elif from_type == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0

        elif from_type == np.float32:
            audio_data = audio_data

        else:
            return None

        return audio_data

    def save_audio_to_wav(self, audio_data: np.ndarray) -> str:
        """_summary_

        Args:
            audio_data (np.ndarray): _description_

        Returns:
            str: _description_
        """
        # ファイル名を生成
        filename = "output_audio_" + str(int(np.random.rand() * 10000)) + ".wav"
        output_path = os.path.join(self.output_dir, filename)

        # Waveファイルの書き出し
        with wave.open(output_path, 'w') as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)
            wf.setframerate(self._rate)
            byte_data = struct.pack('<' + 'h' * len(audio_data), *audio_data)
            wf.writeframes(byte_data)

        print(f"\n Saved: {output_path}")

        return output_path

    def audio_cb(self, msg: AudioStamped) -> None:

        if not self.enabled:
            return

        self._format = msg.audio.info.format
        self._channels = msg.audio.info.channels
        self._rate = msg.audio.info.rate
        self._chunk = msg.audio.info.chunk

        # audio_array = self.int2float(msg_to_array(msg.audio))
        audio_array = msg_to_array(msg.audio)

        # if audio_array is None:
        #     self.get_logger().error(f"Format {msg.audio.info.format} unknown")
        #     return
        # speech_dict = self.vad_iterator(audio_array)

        # if speech_dict:
        #     self.get_logger().info(str(speech_dict))

        #     if not self.recording and "start" in speech_dict:
        #         self.recording = True
        #         self.data = []

        #     elif self.recording and "end" in speech_dict:
        #         self.recording = False
        #         self.data.extend(audio_array.tolist())

        #         if len(self.data) / msg.audio.info.rate < 1.0:
        #             pad_size = (
        #                 msg.audio.info.chunk + msg.audio.info.rate - len(self.data)
        #             )
        #             self.data.extend(pad_size * [0.0])

        #         vad_msg = Float32MultiArray()
        #         vad_msg.data = self.data
        #         self._pub.publish(vad_msg)

        if self.recording:
            self.data.extend(audio_array.tolist())
            print(f"\r recording ... {len(self.data) / msg.audio.info.rate}[sec] / {self.record_duration}[sec]", end="")
            if len(self.data) / msg.audio.info.rate > self.record_duration:
                self.save_audio_to_wav(self.data)
                self.data = []

    def enable_cb(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        res.success = True
        self.enabled = req.data

        if self.enabled:
            res.message = "Silero enabled"
            self.vad_iterator.reset_states()
        else:
            res.message = "Silero disabled"
            self.recording = False
            self.data = []

        self.get_logger().info(res.message)
        return res


def main():
    rclpy.init()
    node = SileroVadNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
