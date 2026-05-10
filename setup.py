from setuptools import setup, find_packages

setup(
    name="video-stego-flow",
    version="0.1.0",
    description="视频帧间信息隐藏实验平台 (Video Steganography Experiment Platform)",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python-headless>=4.7.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
    ],
    entry_points={
        "console_scripts": [
            "video-stego=video_stego.cli:main",
        ],
    },
)
