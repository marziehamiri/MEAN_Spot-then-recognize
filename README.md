
# Facial Expression Spotting and Recognition Improvement in Long Videos

![](https://raw.githubusercontent.com/marziehamiri/MEAN_Spot-then-recognize/main/images/spotrecog.jpg)


## Overview
This project introduces a dual-network architecture for temporal phase detection and emotion recognition based on facial expressions.

Two independent convolutional networks were trained:

Face-based network — captures global facial expression features.
Eyebrow-region network — focuses on subtle eyebrow movements crucial for emotional expression.
To enhance accuracy, a late fusion strategy combines the probabilistic outputs of both networks, merging global and local information for more stable emotion prediction.

Each network contributes with a weighted linear combination (70% face, 30% eyebrow) at both temporal detection and emotion classification stages.

Training setup
Emotion recognition network: 200 epochs
Temporal interval detection network: 40 epochs
Results (CAS(ME)² Dataset)
After late fusion:

Accuracy for “other” class increased from 0% → 16%

Accuracy for “negative” class improved from 53% → 58%

The model shows reduced semantic errors and balanced predictions across emotion categories.

This late-fusion approach significantly improves robustness and overall performance compared to single-network baselines.


---

## Pretrained Weights

You can download the pretrained model weights from the following link:



---

## Author / Credits

This project is based on the original code by **genbing67**
Email: [genbing67@gmail.com](mailto:genbing67@gmail.com)

All modifications and enhancements in this repository were made by the current contributor.
