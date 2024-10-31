#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import rclpy
from rclpy.node import Node
from rclpy.time import Duration
from std_srvs.srv import SetBool
from rclpy.action import ActionClient
from speech_recognition_msgs.action import SpeechRecognition


class RecognitionUtils(Node):

    def __init__(self) -> None:
        super().__init__("speech_recognition_utils")
        self._action_client = ActionClient(self, SpeechRecognition, 'speech_recognition_action')

    def _send_goal(self, recognition_type: str, language="en", dictionary="", whisper_prompt="", ):
        goal_msg = SpeechRecognition.Goal()
        goal_msg.recognition_type = recognition_type
        goal_msg.language = language
        goal_msg.dictionary = dictionary
        goal_msg.whisper_prompt = whisper_prompt

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, self._get_result_future)
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback.status_message} (Current distance: {feedback_msg.feedback.current_distance})')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.recognition_result}')
        rclpy.shutdown()

    def run(self):
        self._send_goal("whisper")


def main():
    rclpy.init()
    recognition_utils = RecognitionUtils()
    recognition_utils._send_goal("whisper")
    rclpy.spin(recognition_utils)


if __name__ == "__main__":
    main()
