# 视频帧间信息隐藏实验平台

`video-stego-flow` 是一个信息隐藏与网络空间安全方向的可运行开源项目，包含核心算法代码、命令行入口、实验配置、示例脚本和 smoke tests。

## Overview

本项目围绕视频载体中的信息隐藏问题，提供视频解码、帧序列采样、运动区域分析、秘密消息分片、嵌入位置选择和鲁棒性测试等功能。平台支持对帧内像素域、频域系数和帧间冗余区域进行实验对比，并可模拟转码、裁剪、抽帧、压缩等常见视频处理攻击。系统输出主观样例和客观指标，便于研究视频水印、短视频版权保护和多媒体取证中的信息隐藏问题，也便于保存实验样例、复现实验参数和对比图表。

## Features

- 统一的数据加载、实验配置和结果保存流程
- 面向信息隐藏/数字水印/隐写分析任务的模块化设计
- 支持实验指标输出、样例结果归档和后续算法扩展
- 适合课程实验、毕业设计、论文复现实验和课题组日常开发

## Quick Start

```bash
python examples/demo.py
python -m unittest discover -s tests
python -m video_stego_flow.cli --message "demo payload" --report docs/cli_report.md
```

## Keywords

video watermark · frame sampling · robustness · multimedia forensics

## Authors

- 负责人：林裕斌
- 参与人：曾科、田承金
- 指导教师：吕善翔
- 单位：暨南大学网络空间安全学院

## License

本项目建议采用 MIT License；实际开源时请根据课题组要求确认许可证。
