Project Overview
This project aims to develop a robust system for analyzing the impact of advertisement placements in real time. The system tracks the number of humans within a specified area, particularly noting those who are actively looking at advertisements on both traditional and modern billboards. By using advanced computer vision technologies, including YOLO for human detection and head pose estimation, the system determines the focus of individuals on advertisements. This technology provides invaluable insights into advertisement effectiveness, enabling advertisers to measure engagement and reach of their campaigns, thereby supporting data-driven negotiations on billboard pricing, which traditionally relies on estimated footfall data alone.

Training Data
The models were trained using a dataset of 1,000 videos, each 6 seconds long, showcasing individuals looking at various angles toward and away from a billboard. This dataset annotates the duration of gaze toward and away from the billboard, assessing whether the duration of looking at the advertisement surpasses a minimal threshold sufficient for reading the content, thereby marking effective engagement.

To ensure precise accuracy, different types of noise were added to the original 1,000 videos, expanding the dataset to a unique collection of 7,000 videos for training purposes. This enhancement helps in improving the robustness and performance of the models under varied environmental conditions.

Dataset Restrictions
The training dataset is proprietary, compiled from floor CCTV footage of 1,000 university students. It was created with the permission of the university and remains its property. Due to this, the dataset is not publicly available.

Code Description
The codebase includes implementations for real-time video processing and human detection. It features video capture, YOLO-based person detection, head pose estimation, and engagement metrics calculation based on the orientation and duration of a person's gaze toward the advertisement. The design is tailored to adapt to different datasets while maintaining precision in detecting and assessing human engagement.

This README is designed to provide a clear understanding of the project's scope and the implications for advertising strategy in both traditional and modern advertising conte