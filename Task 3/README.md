```markdown
# Conditional Image Colorization 🖌️🖼️

This project enables **interactive colorization of grayscale images** based on **user-defined conditions** such as "make the sky blue" or "make the grass green." Users can select image regions and assign custom colors through an intuitive Streamlit interface.

> ⚠️ **Note:** This project does **not use pre-trained weights or a saved model file**. Instead, colorization is done directly based on user inputs — no automatic predictions are required. The model architecture is initialized in real-time and used only to support region-wise operations.

---

## 🎯 Project Objective

This implementation is built to fulfill the task:
> **"Create a model that colorizes grayscale photos based on user-defined circumstances, such as making the sky blue or the grass green."**

The GUI allows users to:
- Upload grayscale images
- Select rectangular regions
- Assign custom colors (e.g., blue for sky, green for grass)
- Instantly preview and download the colorized result

---

## 🚀 How to Run

1. **Clone the repository** or download the files:
   ```bash
   git clone https://github.com/your-username/conditional-colorizer.git
   cd conditional-colorizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run Task_3.py
   ```

4. **Use the Interface**:
   - Upload a grayscale image
   - Define regions using coordinates
   - Pick colors for each region
   - Click **"Colorize Image"** and preview the result
   - Download the colorized image as PNG
```
