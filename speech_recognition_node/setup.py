import os
import sys
import glob
from setuptools import find_packages, setup

package_name = 'speech_recognition_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share/' + package_name, "io/audio"), glob.glob("io/audio/*.wav")),
        (os.path.join('share/' + package_name, "io/audio"), glob.glob("io/audio/*.mp3")),
        (os.path.join('share/' + package_name, "io/output"), glob.glob("io/output/*.wav")),
        (os.path.join('share/' + package_name, "io/prompts"), glob.glob("io/prompts/*.py")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='toyota',
    maintainer_email='yano.yuuga158@mail.kyutech.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'audio_recorder_node = speech_recognition_node.audio_recorder_node:main',
            'speech_recognition_node = speech_recognition_node.speech_recognition_node:main',
        ],
    },
)
