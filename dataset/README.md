# Dataset Information

## Dataset Description

The dataset used in this project was compiled from multiple publicly available traffic surveillance video sources. Since no single dataset contained all the required accident scenarios, videos were collected from different sources and organized into a balanced dataset suitable for road accident detection.

The compiled dataset contains traffic surveillance videos representing both normal driving conditions and road accident scenarios captured under various environmental conditions.

## Dataset Details

- **Dataset Type:** Traffic Surveillance Videos
- **Total Videos Used:** 401
- **Vehicle Categories:** Cars, Bikes, Trucks, Buses, and Other Road Vehicles
- **Application:** Road Accident Detection, Multi-Object Tracking, Collision Detection, and Severity Classification

## Dataset Preparation

The collected videos were preprocessed before model implementation by:
- Extracting video frames using OpenCV
- Resizing frames for YOLOv8 input
- Organizing accident and non-accident samples
- Performing vehicle detection and tracking
- Generating collision and severity analysis outputs

## Dataset Availability

The complete dataset is **not included** in this repository because:
- It was compiled from multiple publicly available sources.
- The dataset size exceeds GitHub's storage limits.
- Some source videos have individual licensing and redistribution restrictions.

Users interested in reproducing this work may use publicly available traffic surveillance and accident video datasets from platforms such as:
- Kaggle (https://www.kaggle.com/)
- Roboflow Universe (https://universe.roboflow.com/)
- Public traffic surveillance videos released for research purposes.
