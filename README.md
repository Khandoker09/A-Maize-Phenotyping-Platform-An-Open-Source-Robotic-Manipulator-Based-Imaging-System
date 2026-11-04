# Universal-Phenotyping-Acquisition-Platform-UPAP-

Universal Phenotyping Acquisition Platform

## System flow 

![UPAP system flow](fig2.jpg)

# Plant Point Cloud Stem–Leaf Segmentation Using SimplePointNetSeg

This repository provides a lightweight deep-learning workflow for organ-level segmentation of plant point clouds. The pipeline uses manually labelled 3D point cloud parts to train a simple PointNet-style semantic segmentation model, named `SimplePointNetSeg`, for classifying each point as either stem or leaf.

The workflow includes point cloud labelling, conversion of labelled plant organs into training files, visual label checking, model training, stem–leaf prediction, individual leaf separation, and basic trait extraction. Each point is represented using spatial, colour, and normal information: `x, y, z, r, g, b, nx, ny, nz`.

The `SimplePointNetSeg` model performs point-wise semantic segmentation by learning local point features and combining them with a global plant-level feature. After stem and leaf prediction, classical geometry-based processing is used to separate individual leaves, order them from bottom to top, and estimate plant traits such as plant height, leaf length, and leaf angle.

## Workflow

1. Manually separate stem and individual leaves from each plant point cloud.
2. Convert labelled parts into training files with semantic and instance labels.
3. Visualize labelled data to verify stem and leaf annotations.
4. Train the `SimplePointNetSeg` model for stem–leaf semantic segmentation.
5. Predict stem and leaf points on new plant point clouds.
6. Split predicted leaf points into individual leaves using clustering.
7. Estimate plant height, leaf length, and leaf angle from the segmented point cloud.
This repository is intended as a practical starting point for 3D plant phenotyping using low-complexity deep learning and geometry-based post-processing.
