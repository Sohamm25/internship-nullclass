# Cross-Domain Image Colorization

A Streamlit app that colorizes grayscale images across multiple domains—sketches, infrared, sepia, cyanotype, satellite and anaglyph 3D—using computer vision algorithms.

**Model Training File:** **THIS PROJECT USES DEEP LEARNING AND COMPUTER VISION TECHNIQUES, SO NO SEPARATE MODEL TRAINING FILE IS INCLUDED.**

## Features
- Convert black-and-white sketches into crisp color outputs  
- Map infrared/grayscale intensity to a thermal-style palette  
- Apply classic sepia or cyanotype effects  
- Visualize satellite-style terrain colors  
- Generate red-cyan anaglyph 3D images  
- Adjustable “Artistic Effect” slider for extra contrast/saturation  

## Requirements
- Python 3.7+  
- [Streamlit](https://streamlit.io/)  
- numpy  
- opencv-python  
- pillow  
- matplotlib  
- scikit-image  
- scipy  

## Installation
1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/cross-domain-colorization.git
   cd cross-domain-colorization
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Usage
Run the Streamlit app:

bash
Copy
Edit
streamlit run Task_5.py
Upload your grayscale image (JPG/PNG).

Select the colorization domain.

Adjust the “Artistic Effect” slider (optional).

Click Process Image and download your resul
