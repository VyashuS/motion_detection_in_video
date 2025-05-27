# 🎥 Motion Detection in Video using OpenCV

This project demonstrates a real-time **motion detection system** using OpenCV in Python. It identifies movement in a video stream from the webcam and highlights the detected areas using contour detection and bounding boxes.

---

## 📌 Features

- Real-time video capture via webcam
- Background subtraction for detecting motion
- Foreground mask generation and refinement
- Noise removal using erosion
- Contour detection of moving objects
- Bounding boxes and visual alerts on motion detection

---

## 🛠️ Techniques Used

- `cv2.createBackgroundSubtractorMOG2()` – background subtraction  
- `cv2.threshold()` – for binary mask creation  
- `cv2.erode()` – morphological operation to reduce noise  
- `cv2.findContours()` – detects contours in the binary mask  
- `cv2.boundingRect()` – gets bounding rectangle around the contour  
- `cv2.rectangle()` – draws bounding box on moving object  
- `cv2.putText()` – displays alerts like "Intrusion Detected"

---

## 🔧 Requirements

- Python 3.x  
- OpenCV  
- NumPy  

Install dependencies using:

```bash
pip install opencv-python numpy
