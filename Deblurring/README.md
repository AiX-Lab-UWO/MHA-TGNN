# DGW Landmark Extraction After Deblurring

This repository provides a full preprocessing pipeline that **deblurs all DGW dataset images using the MB-TaylorFormerV2 model** and then extracts **478 3D facial landmarks** from each deblurred image using **MediaPipe FaceMesh**.

The result is two landmark CSVs (for training and validation sets) that can be used as inputs to graph-based gaze estimation or facial analysis models.

---

## 📘 Overview

The pipeline performs the following sequence for every image in the DGW dataset:

1. **Image Deblurring**  
   - Uses **MB-TaylorFormerV2**, a Transformer-based image restoration model.  
   - Loaded directly from the local cloned repository with pretrained weights (`gopro-XL.pth`).  
   - Automatically handles padding to multiples of 8 for correct model input dimensions.

2. **Facial Landmark Extraction**  
   - Applies **MediaPipe FaceMesh** (`static_image_mode=True`, `refine_landmarks=True`)  
   - Detects **478 3D landmarks**: `(x, y, z)` coordinates for each facial keypoint.

3. **Data Export**  
   - Writes results into CSV files with one row per image and columns:  
     ```
     filename, x0, y0, z0, x1, y1, z1, …, x477, y477, z477, Label
     ```
   - Failed samples (e.g., unreadable images, no detected face) are logged into text files.

4. **Timing Metrics**  
   - Measures and prints:
     - Deblurring time (ms) for the first successfully processed image.
     - Landmark extraction time (ms) for that image.
     - Total wall-clock processing time.
