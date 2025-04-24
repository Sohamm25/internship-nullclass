# Selective Image Colorization using Semantic Segmentation 🎨🐾

This project allows users to selectively colorize specific regions (foreground, background, or entire image) using semantic segmentation. A custom-trained U-Net model is used for segmenting cats and dogs, and a separate model colorizes the grayscale image based on the selected region.

> ⚠️ **Note:** The segmentation model was trained on the [Oxford-IIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), so for best results, use images of **cats and dogs** only.

---

## 🚀 How to Run

1. **Clone the repository** or download the files:
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Place your trained U-Net model** in the project directory:
    ```
    my_unet_model.h5
    ```

4. **Run the Streamlit app**:
    ```bash
    streamlit run Task_2.py
    ```

5. **Upload an image** (preferably of a cat or dog), choose the region to colorize, and preview or download the result!

---

## ✨ Features

- Semantic segmentation using a custom U-Net model.
- Region-based colorization (foreground, background, or entire image).
- Clean Streamlit-based user interface.
