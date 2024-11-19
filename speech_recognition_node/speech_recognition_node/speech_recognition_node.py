#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import sys
import time
import json
import wave
import random
import shutil
import pprint
import pyaudio
import openai
from rclpy.executors import MultiThreadedExecutor

# silero_vad
import torch
torch.set_num_threads(1)
import numpy as np
import collections
import torchaudio
torchaudio.set_audio_backend("soundfile")

from std_msgs.msg import String, Float32
from audio_common_msgs.msg import AudioData
from speech_recognition_msgs.action import SpeechRecognition

from std_srvs.srv import SetBool

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Duration
from rclpy.action import ActionServer

from audio_common.utils import msg_to_array, msg_to_data
from audio_common_msgs.msg import AudioStamped
from ament_index_python.packages import get_package_share_directory

# # 音声強調に必要なモジュール
# sys.path.append(roslib.packages.get_pkg_dir("tam_speech_recog"))
# from include.sgmse.sgmse.model import ScoreModel
# from include.sgmse.sgmse.util.other import pad_spec
# import torch
# from soundfile import write
# from torchaudio import load
# from os.path import join


class SpeechRecognitionServer(Node):
    """
    音声認識を行うクラス
    アクションサーバによって実装している
    """

    def __init__(self) -> None:
        """
        マイクの設定とrosインタフェースの初期化
        """
        super().__init__("speech_recognition_node")
        self.action_signal = False

        ###################################################
        # ROSPARAMの読み込み
        ###################################################

        self.declare_parameter("speech_recognition/sampling_rate", 16000)
        self.declare_parameter("speech_recognition/channels", 1)
        self.declare_parameter("speech_recognition/sample_width", 2)
        self.declare_parameter("speech_recognition/chunk", 512)
        self.declare_parameter("speech_recognition/output", False)
        self.declare_parameter("speech_recognition/vad_aggressiveness", 3)
        self.declare_parameter("speech_recognition/save_as", True)
        self.declare_parameter("speech_recognition/whisper_model", "medium")
        self.declare_parameter("speech_recognition/time_out", 10)
        self.declare_parameter("speech_recognition/beep_sound", True)

        self.sampling_rate = self.get_parameter("speech_recognition/sampling_rate").value
        self.channels = self.get_parameter("speech_recognition/channels").value
        self.sample_width = self.get_parameter("speech_recognition/sample_width").value
        self.chunk = self.get_parameter("speech_recognition/chunk").value
        self.output_flag = self.get_parameter("speech_recognition/output").value
        self.vad_aggressiveness = self.get_parameter("speech_recognition/vad_aggressiveness").value
        self.flag_save_as = self.get_parameter("speech_recognition/save_as").value
        self.whisper_model_type = self.get_parameter("speech_recognition/whisper_model").value
        self.time_out = self.get_parameter("speech_recognition/time_out").value
        self.beep_sound = self.get_parameter("speech_recognition/beep_sound").value

        self.output_dir = os.path.join(get_package_share_directory('speech_recognition_node'), 'io/output')

        # ros interface
        self._audio_sub = self.create_subscription(AudioStamped, "audio", self.audio_cb, qos_profile_sensor_data)

        # TODO(yano) Audio messageの配信ノードの動作開始/ストップの制御
        # self.srv_audio_publisher = self.create_client(SetBool, '/audio_publisher/run_enable')
        # while not self.srv_audio_publisher.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Service /audio_publisher/run_enable is not available, waiting...')

        # rospy.wait_for_service('/audio_publisher/run_enable')
        # self.srv_audio_publisher = rospy.ServiceProxy('/audio_publisher/run_enable', SetBool)

        if self.flag_save_as:

            # 現在の日付と時刻を取得してフォルダを作成
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.audio_data_dir = os.path.join(get_package_share_directory('speech_recognition_node'), f"io/output/{timestamp}/")

            os.makedirs(self.audio_data_dir)
            self.get_logger().info("\"save as\" is True")
            self.get_logger().info(("create new dir: " + self.audio_data_dir))

            # インクリメントする変数を初期化
            self.save_counter = 1
            self.audio_path = self.audio_data_dir + str(self.save_counter) + ".wav"

            # 認識結果を保存するためのcsvファイルを用意する
            self.csv_path = self.audio_data_dir + "result.csv"
            self.get_logger().info(("create new csv: " + self.csv_path))

            # csvファイルのヘッダを作成
            csv_header = ["ファイル名", "辞書", "認識モデル", "認識結果（置換前）"]
            with open(self.csv_path, "w", newline='\n') as file:
                writer = csv.writer(file)
                writer.writerow(csv_header)

        else:
            self.get_logger().info("\"save as\" is False")
            self.audio_data_dir = os.path.join(get_package_share_directory('speech_recognition_node'), f"io/output")
            self.audio_path = os.path.join(self.audio_data_dir, "audio.wav")
            # self.audio_path = roslib.packages.get_pkg_dir("tam_speech_recog") + "/io/audiodata/audio.wav"

        # 音声データを保存するリスト
        self.voiced_frames = []
        self.current_frame = None

        # voice activity detectionの初期化
        # self.vad = webrtcvad.Vad(self.vad_aggressiveness)

        # アクションサーバからの信号をもとにしたフラグ
        self.recognition_result = None
        self.recognition_type = "whisper"

        # 音声保存処理に関するフラグなど
        self.triggered_start = False
        self.triggered_end = False

        # whisperのプロンプト辞書読み込み
        if self.recognition_type == "whisper":
            pass
            # self.promt_dicts = self.load_prompt()
        else:
            # voskにおける辞書の読み込み
            self.dicts = self.load_dictionary()

        self.padding_duration_ms = 10
        self.frame_duration_ms = 10
        self.num_padding_frames = 20
        # We use a deque for our sliding window/ring buffer.
        self.ring_buffer = collections.deque(maxlen=self.num_padding_frames)
        self.ring_buffer_w_past = collections.deque(maxlen=(self.num_padding_frames)*2)

        # whisperの認識モデル初期化
        # self.model_whisper = whisper.load_model(self.whisper_model_type)

        # voskの認識モデル初期化
        # self.model_vosk = Model(lang="en-us")

        # pyaudioを初期化
        # self.p = pyaudio.PyAudio()

        # ストリームを開始
        # self.stream = self.p.open(
        #     format=self.p.get_format_from_width(self.sample_width),
        #     channels=self.channels,
        #     rate=self.sampling_rate,
        #     output=True,
        #     frames_per_buffer=self.chunk
        # )

        # 話している時間のタイムアウト処理のための変数初期化
        self.record_start_time = self.get_clock().now()

        # Image pub
        # self.image = rospy.Publisher('/show_image/data', Image, queue_size=10)
        # self.bridge = CvBridge()

        # # 音声強調に必要なパラメタ
        # self.corrector_steps = 1
        # self.corrector_cls = "ald"
        # self.N = 30
        # self.snr = 0.5
        # # self.sr = 16000
        # self.sgmse_checkpoint_file = self.audio_path = roslib.packages.get_pkg_dir("tam_speech_recog") + "/io/checkpoint/train_wsj0_2cta4cov_epoch=159.ckpt"

        # # Load score model
        # self.model = ScoreModel.load_from_checkpoint(self.sgmse_checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
        # self.model.eval(no_ema=False)
        # self.model.cuda()

        self.silero_vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True
        )

        self.api_key = os.environ.get("OPENAI_API")
        openai.api_key = self.api_key
        self.check_authenticate()

        # 音声認識部をアクションサーバとして実装
        # self.action_server_register("speech_recog", "tam_speech_recognition", SpeechRecognitionAction, self.cb_recognition)
        self._sr_action_server = ActionServer(self, SpeechRecognition, 'speech_recognition_action', self.cb_recognition)

        self.get_logger().info("ready to speech recognition")

        self.count = 0
        self.num_ave = 0

        self.whisper_prompt = ""
        self.language = ""
        self.dictionary = ""
        self.feedback_signal = False
        self.fillers = ["えーっと．", "うーんと．", "えっとね．"]

    def check_authenticate(self) -> bool:
        """OpenAIのAPIキーが認証されているかを検証する関数

        Returns:
            bool: 認証が通ればTrue, 通らなければFalse
        """
        try:
            _ = openai.models.list()
        except openai.AuthenticationError:
            self.get_logger().error("OPENAI_API authenticate is failed")
            return False
        except Exception as e:
            self.get_logger().error(f"{e}")
            return False
        self._logger.info("OPENAI API authenticate is success")
        return True

    # def load_dictionary(self, show_dicts=True) -> dict:
    #     """
    #     voskでの音声認識で使用できるような形に辞書を整形する関数
    #     """
    #     # 音声認識の辞書の作成
    #     dicts = {}
    #     # dictionary.pyから読み込んだ辞書型変数を基に，音声認識用の辞書を作成
    #     for dict_name, dict_values in GP_DICTIONARY.items():
    #         temp_dictionary = "[\""

    #         for dict_value in dict_values:
    #             temp_dictionary = temp_dictionary + dict_value + " "

    #         temp_dictionary = temp_dictionary[:-1] + "\"]"  # 最後のスペースを削除した上で，voskに合わせて整形
    #         temp_dictionary_unk = temp_dictionary[:-2] + "\", \"[unk]\"]"  # unknownを追加した辞書も自動で作成

    #         dict_name_with_unk = dict_name + "_unk"
    #         dicts[dict_name] = temp_dictionary  # 作成した音声認識用の辞書を辞書型配列に保存
    #         dicts[dict_name_with_unk] = temp_dictionary_unk

    #     if show_dicts:
    #         self.get_logger().info("辞書一覧")
    #         pprint.pprint(dicts)  # 確認用

    #     return dicts

    def load_prompt(self, show_dicts=True) -> dict:
        """
        whisperでの音声認識で使用できるような形にプロンプト辞書を整形する関数
        """
        # 音声認識の辞書の作成
        prompt_dicts = {}
        # dictionary.pyから読み込んだ辞書型変数を基に，音声認識用の辞書を作成
        for dict_name, dict_values in GP_WHISPER_SENTENCE.items():
            prompt_dicts[dict_name] = ", ".join(dict_values)

        if show_dicts:
            self.get_logger().info("辞書一覧")
            pprint.pprint(prompt_dicts)  # 確認用

        return prompt_dicts

    def audio_publish_start(self) -> None:
        """
        オーディオデータのパブリッシュを開始する関数
        """
        try:
            self.srv_audio_publisher(True)
            self.get_logger().info("start audio publisher!!")
        except Exception as e:
            self.get_logger().warn("cannot start audio publisher")
            self.logerr(e)

    def audio_publish_stop(self) -> None:
        """
        オーディオデータのパブリッシュを停止する関数
        """
        try:
            self.srv_audio_publisher(False)
        except Exception as e:
            self.get_logger().warn("cannot stop audio publisher")
            self.logerr(e)

    def cb_recognition(self, goal_handle: SpeechRecognition.Goal) -> SpeechRecognition.Result:
        """
        音声認識のアクションが呼び出されたときのコールバック関数
        """
        # サービスを用いて，マイクデータのpublishを開始する
        self.time_out = self.get_parameter("speech_recognition/time_out").value

        # TODO(yano) 音声データの受信を開始
        # self.audio_publish_start()

        self.action_signal = True
        self.recognition_type = goal_handle.request.recognition_type
        self.dictionary = goal_handle.request.dictionary
        self.whisper_prompt = goal_handle.request.whisper_prompt
        self.language = goal_handle.request.language

        self.get_logger().info(f"recognition_type: {self.recognition_type}")
        self.get_logger().info(f"dictionary: {self.dictionary}")
        self.get_logger().info(f"whisper_prompt: {self.whisper_prompt}")
        self.get_logger().info(f"language: {self.language}")

        # 音声認識の結果が出るまで待機
        # self.action_signalは音声認識結果が得られた時点でFalseに変わる
        start_time = self.get_clock().now()
        self.feedback_signal = False
        while self.action_signal:
            if (self.get_clock().now() - start_time) > Duration(seconds=30):  # 30
                self.action_signal = False
                self.beep_sound = True
                # self.ring_buffer.clear()
                self.get_logger().info("タイムアウトによる音声認識の強制終了")
            self.get_clock().sleep_for(Duration(seconds=1.5))

            if self.feedback_signal:
                self.feedback_signal = False
                feedback = SpeechRecognition.Feedback()
                feedback.status = random.choice(self.fillers)
                goal_handle.publish_feedback(feedback)

        # 認識結果をクライアントに送信
        result = SpeechRecognition.Result()
        result.recognition_result = self.recognition_result
        result.temperature = self.temperature
        result.no_speech_prob = self.no_speech_prob
        result.language = self.language
        goal_handle.succeed()

        # TODO(yano) 音声データのパブリッシュを停止
        # self.audio_publish_stop()
        return result

    def delete(self) -> bool:
        """
        デストラクタ
        一度も音声データを保存していない場合は，作成したフォルダを削除する
        """
        # 一度も保存処理が行われていない場合は，ディレクトリを削除する
        if self.save_counter == 1:
            shutil.rmtree(self.audio_data_dir)
            return True
        return False

    # Provided by Alexander Veysov
    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / 32768
        sound = sound.squeeze()  # depends on the use case
        return sound

    def is_speech_by_silero_vad(self, audio_chunk: bytes, th=0.5) -> bool:
        """
        silerovadによる発話検出
        Args:
            audio_chunk(bytes): 音声データ
            th(float): この値以上で話し始めと判定
        Returns:
            bool 話していると検知した場合True
        """
        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        audio_float32 = self.int2float(audio_int16)

        # get the confidences and add them to the list to plot them later
        confidence = self.silero_vad_model(torch.from_numpy(audio_float32), 16000).item()
        self.get_logger().info(f"confidence: {confidence}")

        if confidence > th:
            return True
        else:
            return False

    def audio_cb(self, msg: AudioStamped) -> None:
        """
        オーディオデータを受信し，喋っている区間のみの録音を行う関数
        """

        # considering_frame = rospy.get_param(rospy.get_name() + "/considering_frame", 10)
        self.considering_frame = 0

        # アクションから音声認識開始の信号を受信するまで処理を行わない
        if self.action_signal is False:
            return

        # beep音をTrueの場合鳴らす
        # if self.beep_sound:
        #     # playsound(roslib.packages.get_pkg_dir("tam_speech_recog") + '/io/sound/beep_sound.mp3')
        #     # Image pub
        #     img_path = roslib.packages.get_pkg_dir("tam_speech_recog") + "/io/image/listening.png"
        #     img = cv2.imread(img_path)
        #     img_msg = self.bridge.cv2_to_imgmsg(img)
        #     self.image.publish(img_msg)

        # オーディオデータの受信を開始
        # feedback = SpeechRecognitionFeedback(status=String("listening (not recording)"))
        # self.action.server.speech_recog.publish_feedback(feedback)
        # self.sr_server.publish_feedback(feedback)
        # self.beep_sound = False

        self._format = msg.audio.info.format
        self._channels = msg.audio.info.channels
        self._rate = msg.audio.info.rate
        self._chunk = msg.audio.info.chunk

        self.current_frame = msg_to_data(msg.audio)
        if self.output_flag:
            pass
            # self.stream.write(self.current_frame)

        # is_speech = self.vad.is_speech(self.current_frame, self.sampling_rate)  # vad
        is_speech = self.is_speech_by_silero_vad(self.current_frame)  # silero-vad

        # 話はじめの検知
        # リングバッファに音声をためつつ，話し始めるまで検証を続ける
        # 話し始めを検出したら，リングバッファに溜まっている音声を含めて音声ファイルを作成開始し，話し終わりの検知へ移行
        if self.triggered_start is False and self.triggered_end is False:
            self.ring_buffer.append((self.current_frame, is_speech))
            self.ring_buffer_w_past.append((self.current_frame, is_speech))
            num_voiced = len([f for f, speech in self.ring_buffer if speech])

            # 話しているフレームの割合が一定数を超えたら，話しはじめたと検出する
            if num_voiced > 0.6 * self.ring_buffer.maxlen:
                self.triggered_start = True

                # これまでのリングバッファに入っているデータを保存ように取り出す
                for f, s in self.ring_buffer_w_past:
                    # self.voiced_frames.append(f)
                    self.voiced_frames.append(f)

                # self.ring_buffer.clear()  # 不要なリングバッファに入っているデータを削除
                # self.ring_buffer_w_past.clear()  # 不要なリングバッファに入っているデータを削除

                self.get_logger().info("start recording")

                # 話しはじめの時間を保持（タイムアウト処理のため）
                self.record_start_time = self.get_clock().now()
                # self.record_start_time = self.get_clock().now()

        # 話終わりの検知
        # 一定フレーム以上話していないと検出したら，録音を停止
        elif self.triggered_start is True and self.triggered_end is False:
            self.voiced_frames.append(self.current_frame)
            self.ring_buffer.append((self.current_frame, is_speech))
            # num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
            num_unvoiced = len([f for f, speech in self.ring_buffer if speech])

            # 話しているフレームの割合が一定数を下回ったら，話し終えたと検出する
            # 少しでも閾値以下の声だったら終了していたが，閾値を下回って５０回先の結果まで確認して終了するようにする
            # 雑だからもう少しちゃんとやる
            if num_unvoiced < 0.01 * self.ring_buffer.maxlen:
                self.num_ave += num_unvoiced
                self.count += 1
                if self.count > self.considering_frame:
                    if self.num_ave < 0.3 * self.ring_buffer.maxlen:
                        self.triggered_end = True
                        self.count = 0
                # self.triggered_start = False

            if (self.get_clock().now() - self.record_start_time) > Duration(seconds=self.time_out):
                self.triggered_end = True
                # self.triggered_start = False

        # これまでの音声を録音し，音声ファイルを作成
        # 音声認識までを行い，次の入力に備える
        if self.triggered_end:
            # 別名保存するモードの場合
            if self.flag_save_as:
                self.audio_path = self.audio_data_dir + str(self.save_counter) + ".wav"
                self.get_logger().debug("save wave file as" + self.audio_path)
                self.save_counter += 1

            self.get_logger().info("stop recording")
            self.feedback_signal = True
            self.make_wave_file(self.audio_path)

            # アクションから与えられた信号をもとに，音声認識のモデルなどを変更するchange
            if self.recognition_type == "vosk":
                self.recognition_result = self.speech_recog_vosk(self.audio_path, self.dictionary)
            elif self.recognition_type == "whisper":
                self.recognition_result, self.temperature, self.no_speech_prob, self.language = self.speech_recog_whisper(self.audio_path, self.whisper_prompt, self.language)
            else:
                self.recognition_result, self.temperature, self.no_speech_prob, self.language = self.speech_recog_whisper(self.audio_path, self.whisper_prompt, self.language)

            current_result = [str(self.save_counter - 1), self.dictionary, self.recognition_type, self.whisper_prompt, self.recognition_result]

            with open(self.csv_path, "a", newline='\n') as file:
                self.get_logger().debug("dump result")
                writer = csv.writer(file)
                writer.writerow(current_result)

            # 次の入力に備える
            self.ring_buffer.clear()  # 不要なリングバッファに入っているデータを削除
            self.ring_buffer_w_past.clear()  # 不要なリングバッファに入っているデータを削除
            self.beep_sound = True
            self.voiced_frames = []
            self.triggered_end = False
            self.action_signal = False
            self.triggered_start = False

    def make_wave_file(self, wave_file_path) -> None:
        """
        音声データを保存するWAVEファイルを作成
        """
        wave_file = wave.open(wave_file_path, "w")
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(self.sample_width)
        wave_file.setframerate(self.sampling_rate)

        # 音声データをWAVEファイルに書き込む
        for data in self.voiced_frames:
            wave_file.writeframes(data)

        # WAVEファイルを閉じる
        wave_file.close()

    def speech_recog_whisper(self, path: str, whisper_prompt: str, language: str) -> str:
        """
        whisperを使った音声認識を行う関数

        Args:
            path (str): 音声認識を行う音声ファイルのパス
            whisper_prompt (str): プロンプト指定
            language(str): 認識言語の指定
        """
        self.stt_model = "whisper-1"
        file = open(path, "rb")
        if language == "":
            self.get_logger().debug("whisperによる音声認識: 言語指定なし")
            if whisper_prompt == "":
                self.get_logger().debug("whisperによる音声認識: プロンプト指定なし")
                transcription = openai.audio.transcriptions.create(model=self.stt_model, file=file, response_format="verbose_json")
                # result = self.model_whisper.transcribe(path, no_speech_threshold=0.7)
            else:
                self.get_logger().debug("whisperによる音声認識: プロンプト指定あり")
                transcription = openai.audio.transcriptions.create(model=self.stt_model, file=file, response_format="verbose_json")
                # result = self.model_whisper.transcribe(path, initial_prompt=whisper_prompt, no_speech_threshold=0.7)
        else:
            self.get_logger().debug("whisperによる音声認識: 言語指定あり")
            if whisper_prompt == "":
                self.get_logger().debug("whisperによる音声認識: プロンプト指定なし")
                transcription = openai.audio.transcriptions.create(model=self.stt_model, file=file, response_format="verbose_json")
                # result = self.model_whisper.transcribe(path, language=language, no_speech_threshold=0.7)
            else:
                self.get_logger().debug("whisperによる音声認識: プロンプト指定あり")
                transcription = openai.audio.transcriptions.create(model=self.stt_model, file=file, response_format="verbose_json")
                # result = self.model_whisper.transcribe(path, language=language, initial_prompt=whisper_prompt, no_speech_threshold=0.7)

        result = transcription
        self.get_logger().info(f"whisper verbose result: {result}")
        self.get_logger().info(f"result.txt type: {type(result.text)}")

        worst_temperature = 0.0
        worst_no_speech_prob = 0.0
        for segment in result.segments:
            if worst_temperature < segment.temperature:
                worst_temperature = segment.temperature
            if worst_no_speech_prob < segment.no_speech_prob:
                worst_no_speech_prob = segment.no_speech_prob

        if result.language == "japanese":
            return_language = "ja"
        elif result.language == "english":
            return_language = "en"
        else:
            return_language = result.language

        return result.text, worst_temperature, worst_no_speech_prob, return_language

    def speech_recog_vosk(self, path: str, dictionary_type: str, return_all=False, show_all_result=False) -> str:
        """voskを用いた音声認識用の関数

        Args:
            path (str): 音声認識を行う音声ファイルのパス
            dictionary_type (str): 辞書指定
            return_all (bool, optional): 認識した生の結果を返り値とする
                返り値がstrの配列となる点に注意
                defaults to False
            show_all (bool, optional): 認識した結果をターミナルに出力する
                defaults to False
        """

        wf = wave.open(path, "rb")
        # 辞書指定が合った場合は，辞書を読み込む
        if dictionary_type == "all":
            # 辞書指定がない場合（デフォルト）
            rec = KaldiRecognizer(self.model_vosk, wf.getframerate())

        else:
            try:
                rec = KaldiRecognizer(self.model_vosk, wf.getframerate(), self.dicts[dictionary_type])
            except Exception as e:
                # 指定された辞書が見つからなかった場合は，辞書無しでの認識を行う
                self.get_logger().trace(f"{e}")
                self.get_logger().warn("cannot use selected dictionary")
                self.get_logger().warn("use dictionary is " + dictionary_type)
                rec = KaldiRecognizer(self.model_vosk, wf.getframerate())

        rec.SetWords(True)
        rec.SetPartialWords(True)
        result_flag = True

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                if show_all_result:
                    final_result = rec.Result()
                    result_flag = False
                else:
                    final_result = rec.Result()
                    result_flag = False
            else:
                pass

        if result_flag:
            final_result = rec.FinalResult()
            if show_all_result:
                # 信頼値なども含めたデータを表示する
                print(final_result)
        else:
            pass

        dict_json = json.loads(final_result)
        recog_txt = dict_json["text"]

        if return_all:
            return dict_json
        else:
            print(recog_txt)
            return recog_txt

    # def speech_enhancement_sgmse(self, path) -> str:
    #     """
    #     sgmseを用いた音声強調を行う関数
    #     Args:
    #         path(str): 対象とする音声ファイルのパス
    #     Rerutns:
    #         path(str): 音声強調後のファイルパス
    #     """
    #     self.get_logger().info("start speech enhancement")

    #     filename = path.split('/')[-1]
    #     self.get_logger().debug(filename)

    #     # Load wav
    #     y, _ = load(path)
    #     T_orig = y.size(1)

    #     # Normalize
    #     norm_factor = y.abs().max()
    #     y = y / norm_factor

    #     # Prepare DNN input
    #     Y = torch.unsqueeze(self.model._forward_transform(self.model._stft(y.cuda())), 0)
    #     Y = pad_spec(Y)

    #     # Reverse sampling
    #     sampler = self.model.get_pc_sampler(
    #         'reverse_diffusion', self.corrector_cls, Y.cuda(), N=self.N,
    #         corrector_steps=self.corrector_steps, snr=self.snr)
    #     sample, _ = sampler()

    #     # Backward transform in time domain
    #     x_hat = self.model.to_audio(sample.squeeze(), T_orig)

    #     # Renormalize
    #     x_hat = x_hat * norm_factor

    #     # Write enhanced wav file
    #     write(join(self.audio_data_dir, filename), x_hat.cpu().numpy(), 16000)

    #     return join(self.audio_data_dir, filename)


def main():
    rclpy.init()
    node = SpeechRecognitionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
