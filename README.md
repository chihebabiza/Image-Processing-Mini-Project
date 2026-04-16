# 🖼️ Image Processing Mini Project (Streamlit)

A graphical image processing application built with **Python + Streamlit** for the 4th Year AI Engineering mini-project.

It allows users to upload images and apply various image processing techniques such as filtering, segmentation, histogram operations, noise addition, and edge detection.

---

## 🚀 Features

### 📥 Image Input
- Upload images in JPG, PNG, JPEG formats
- Display original image

### 🎛️ Image Processing Operations
- Grayscale conversion
- Thresholding
- Histogram Equalization
- Histogram Stretching
- Noise addition (Gaussian, Salt & Pepper)
- Filtering (Blur, Median, Sharpen)
- Edge Detection (Canny)
- Segmentation (K-Means clustering)

### 📊 Visualization
- Histogram display of grayscale image
- Real-time processing updates

### 💾 Export
- Download processed image

---

## 🧰 Technologies Used

- Python
- :contentReference[oaicite:0]{index=0}
- :contentReference[oaicite:1]{index=1}
- NumPy
- Pillow (PIL)

---

## 📁 Project Structure

```

app.py
utils/
│── image_ops.py
│── filters.py
│── segmentation.py
README.md

````

---

## ▶️ Installation & Running

### 1. Install dependencies
```bash
pip install streamlit opencv-python numpy pillow
````

### 2. Run the app

```bash
streamlit run app.py
```

---

## 📌 How It Works

1. Upload an image
2. Select an operation from the sidebar
3. Adjust parameters (if needed)
4. View processed output instantly
5. Download result if required

---

## 🎯 Learning Outcomes

* Basic image processing techniques
* Working with OpenCV in Python
* Building interactive web apps using Streamlit
* Modular project structure in Python

---

## 👨‍🎓 Author

Mini Project – 4th Year AI Engineering
University of Sétif

---

## 📄 License

For academic use only.

