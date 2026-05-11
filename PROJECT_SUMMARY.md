# 视频帧间信息隐藏实验平台

英文名称：`video-stego-flow`

开源地址：`https://github.com/Buzzbbb/video-stego-flow`

项目时间：2025年2月-至今

## 作者信息

- 负责人：林裕斌，专业：网络空间安全，硕士生
- 参与人：曾科，专业：网络空间安全，硕士生
- 参与人：田承金，专业：网络空间安全，硕士生
- 指导教师：吕善翔，网络空间安全学院教师

## 项目内容

本项目围绕视频载体中的信息隐藏问题，提供视频解码、帧序列采样、运动区域分析、秘密消息分片、嵌入位置选择和鲁棒性测试等功能。平台支持对帧内像素域、频域系数和帧间冗余区域进行实验对比，并可模拟转码、裁剪、抽帧、压缩等常见视频处理攻击。系统输出主观样例和客观指标，便于研究视频水印、短视频版权保护和多媒体取证中的信息隐藏问题，也便于保存实验样例、复现实验参数和对比图表。

## 影响力

项目开源后可为视频水印与多媒体取证实验提供基础流程，帮助学生理解视频压缩、帧间冗余和鲁棒嵌入之间的关系。

## 开发语言

Python

## 代码规模

1012行（按当前项目 src/tests/examples 下 Python 代码统计）

## 建议仓库结构

```text
video-stego-flow/
├── README.md
├── LICENSE
├── PROJECT_SUMMARY.md
├── src/
├── examples/
├── tests/
├── docs/
└── screenshots/
```

## 截图材料

- 项目目录截图：`screenshots/directory.png`
- 项目说明截图：`screenshots/readme.png`
- 项目声明截图：`screenshots/license.png`

## 关键词

video watermark, frame sampling, robustness, multimedia forensics
