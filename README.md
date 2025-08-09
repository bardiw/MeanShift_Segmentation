# 🖼️ Mean Shift Image Segmentation

This project implements **Mean Shift Segmentation** using **k-d trees** for efficient nearest neighbor search.  
The algorithm segments an image into regions based on both **spatial** and **color** features.

---

## 📌 Features
- Custom **Mean Shift** implementation (no dependency on OpenCV's `pyrMeanShiftFiltering`)
- Uses **cKDTree** for faster neighbor search
- Works in **LAB color space** for better perceptual segmentation
- Adjustable parameters for:
  - Spatial bandwidth (`h_s`)
  - Range (color) bandwidth (`h_r`)
  - Convergence threshold (`epsilon`)
  - Maximum iterations
- Resizes large images automatically for efficiency
- Saves and displays the segmented result

---

## 📂 Project Structure  
MeanShift_Segmentation/  
│  
├── image sample 1.png # Sample input and output image  
├── image sample 2.png # Sample input and output image  
├── main.py # Main project script  
└── README.md # Documentation  

