# 🖼️ Image Processing App (Streamlit)

A modular **image processing web application** built with **Python, OpenCV, NumPy, and Streamlit**.
It provides a simple UI to apply classical image processing techniques such as filtering, edge detection, thresholding, noise addition, and histogram operations.

---

## 🚀 Features

### 🎨 Image Processing Operations

* Grayscale conversion
* Histogram Equalization
* Histogram Stretching
* Thresholding:

  * Single Threshold
  * Double Threshold
  * Otsu Thresholding
* Noise generation:

  * Gaussian noise
  * Salt & Pepper noise
* Filtering:

  * Mean filter
  * Gaussian filter
  * Median filter
* Edge Detection:

  * Canny
  * Sobel
  * Prewitt
  * Roberts
  * Laplacian
  * Laplacian of Gaussian (LoG)

---

### 📊 Visualization

* Original vs Processed image comparison
* Histogram visualization for intensity distribution

---

### 💾 Export

* Download processed images in PNG format

---

## 🧠 Architecture (Modular Design)

The project is structured in a **clean modular way**:

```
project/
│
├── app.py                  # Main Streamlit application
│
├── ui/                    # UI logic (Streamlit controls)
│   ├── edges.py
│   ├── filters.py
│   ├── noises.py
│   ├── thresholds.py
│   └── histograms.py
│
├── utils/                # Core image processing logic
│   ├── edges.py
│   ├── filters.py
│   ├── noises.py
│   ├── thresholds.py
│   ├── histograms.py
│   └── image_ops.py
│
└── README.md
```

---

## 🛠️ Tech Stack

* Python 🐍
* Streamlit ⚡
* OpenCV 👁️
* NumPy 🔢
* PIL (Pillow) 🖼️

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/chihebabiza/Image-Processing-Mini-Project.git
cd Image-Processing-Mini-Project
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 🧩 How It Works

1. Upload an image
2. Select an operation from the sidebar:

   * Filtering
   * Edge detection
   * Noise addition
   * Thresholding
   * Histogram operations
3. Adjust parameters dynamically
4. View results instantly
5. Download processed image

---

## 📊 Example Workflow

```
Upload Image → Apply Gaussian Noise → Apply Median Filter → Detect Edges → View Histogram
```

---

## 🔥 Key Design Highlights

### ✔ Modular UI separation

Each operation has its own UI handler:

* `apply_edge_ui`
* `apply_filter_ui`
* `apply_noise_ui`
* `apply_threshold_ui`

### ✔ Clean architecture

* UI layer (`ui/`)
* Logic layer (`utils/`)
* Main controller (`app.py`)

### ✔ Scalable design

Easy to add new:

* filters
* edge detectors
* segmentation methods
* ML-based processing

---

## 🚀 Future Improvements

* 🧠 Add segmentation (K-means, watershed)
* ⚡ Parallel processing (NumPy / multiprocessing)
* 🤖 Deep learning filters (denoising autoencoder)
* 📱 REST API backend (FastAPI)
* 🎨 Image pipeline builder (drag & drop operations)

---

## 👨‍💻 Author

Developed as an **Image Processing Lab Project** using Streamlit and OpenCV.

---

## 📜 License

This project is open-source and free to use for educational purposes.


Just tell me 👍
