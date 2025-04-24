# Cross-Domain Image Colorization

A Streamlit app that colorizes grayscale images across multiple domains—sketches, infrared, sepia, cyanotype, satellite and anaglyph 3D—using computer vision algorithms.

# Note: This project uses deep learning and computer vision techniques, so no separate model training file is included.

## Features

- Convert black-and-white sketches into crisp color outputs
- Map infrared/grayscale intensity to a thermal-style palette
- Apply classic sepia or cyanotype effects
- Visualize satellite-style terrain colors
- Generate red-cyan anaglyph 3D images
- Adjustable "Artistic Effect" slider for extra contrast/saturation

## Requirements

- Python 3.7+
- Streamlit
- NumPy
- OpenCV-Python
- Pillow
- Matplotlib
- scikit-image
- SciPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cross-domain-colorization.git
   cd cross-domain-colorization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run Task_5.py
   ```

2. Upload your grayscale image (JPG/PNG)
3. Select the colorization domain
4. Adjust the "Artistic Effect" slider (optional)
5. Click "Process Image" and download your result

## Colorization Algorithms

- **Sketch (black and white)**: Uses adaptive thresholding and edge detection
- **Infrared**: Maps intensity to temperature-like color gradient (blue to red)
- **Sepia tone**: Applies classic brownish-yellow aged photo effect
- **Cyanotype**: Creates blue-toned print effect through channel manipulation
- **Satellite imagery**: Uses intensity-based mapping for terrain visualization
- **Anaglyph 3D**: Creates offset red-cyan channels for stereoscopic effect
