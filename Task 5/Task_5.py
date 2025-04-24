import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import os
import tempfile
from scipy import ndimage
from skimage.segmentation import watershed

def colorize_sketch_bw(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    if np.mean(gray) > 128:
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        binary = dilated
    else:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    if np.mean(binary) < 128:
        binary = 255 - binary
    
    result = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if binary[i, j] > 128:
                result[i, j] = [255, 255, 255]
            else:
                result[i, j] = [0, 0, 0]
                
    return result

def colorize_infrared(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    result = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            intensity = gray[i, j]
            
            if intensity < 64:
                result[i, j] = [0, 0, intensity * 4]
            elif intensity < 128:
                result[i, j] = [0, (intensity - 64) * 4, 255]
            elif intensity < 192:
                result[i, j] = [(intensity - 128) * 4, 255, 255 - (intensity - 128) * 4]
            else:
                result[i, j] = [255, 255 - (intensity - 192) * 4, 0]
    
    return result

def colorize_sepia(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    sepia = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            intensity = gray[i, j]
            sepia[i, j] = [
                min(255, int(intensity * 0.393 + intensity * 0.769 + intensity * 0.189)),
                min(255, int(intensity * 0.349 + intensity * 0.686 + intensity * 0.168)),
                min(255, int(intensity * 0.272 + intensity * 0.534 + intensity * 0.131))
            ]
    
    return sepia

def colorize_cyanotype(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    inverted = 255 - gray
    
    cyanotype = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            intensity = inverted[i, j]
            
            b = min(255, int(60 + intensity * 0.75))
            g = min(255, int(20 + intensity * 0.65))
            r = min(255, int(intensity * 0.35))
            
            cyanotype[i, j] = [r, g, b]
    
    return cyanotype

def colorize_satellite(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    equalized = cv2.equalizeHist(gray)
    
    result = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            value = equalized[i, j]
            
            if value < 64:
                result[i, j] = [128, 51, 0]
            elif value < 128:
                result[i, j] = [0, min(255, 50 + value), 0]
            elif value < 192:
                result[i, j] = [102, 153 - (value - 128), 51]
            else:
                result[i, j] = [min(255, 220 + (value - 192)), min(255, 220 + (value - 192)), min(255, 220 + (value - 192))]
    
    return result

def colorize_anaglyph_3d(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape
    shift = int(w * 0.05)
    
    left = np.zeros((h, w), dtype=np.uint8)
    right = np.zeros((h, w), dtype=np.uint8)
    
    left[:, shift:] = gray[:, :-shift]
    right[:, :-shift] = gray[:, shift:]
    
    anaglyph = np.zeros((h, w, 3), dtype=np.uint8)
    anaglyph[:, :, 0] = left
    anaglyph[:, :, 1] = right
    anaglyph[:, :, 2] = right
    
    return anaglyph

def apply_artistic_filter(img, filter_strength=1.0):
    img_float = img.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + 0.5 * filter_strength)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1.0)

    img_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img_enhanced = np.power(img_enhanced, 0.9 + 0.2 * filter_strength)
    img_enhanced = np.clip(img_enhanced, 0, 1.0)

    return (img_enhanced * 255).astype(np.uint8)

def main():
    st.set_page_config(page_title="Cross-Domain Image Colorization", layout="wide")

    st.title("Cross-Domain Image Colorization")
    st.write("Upload an image to convert it to color based on different domains.")

    st.header("Input Settings")

    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    domain = st.selectbox("Select Colorization Domain:", [
        "sketch (black and white)", 
        "infrared", 
        "sepia tone", 
        "cyanotype",
        "satellite imagery",
        "anaglyph 3D"
    ])

    artistic_effect = st.slider("Artistic Effect Strength", 0.0, 2.0, 1.0, 0.1)

    process_button = st.button("Process Image", use_container_width=True)

    input_image = None
    output_image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        input_image = np.array(image)

        if input_image.shape[-1] == 4:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)

    if input_image is not None and process_button:
        max_dim = 800
        h, w = input_image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            input_image = cv2.resize(input_image, (new_w, new_h))

        domain_map = {
            "sketch (black and white)": colorize_sketch_bw,
            "infrared": colorize_infrared,
            "sepia tone": colorize_sepia,
            "cyanotype": colorize_cyanotype,
            "satellite imagery": colorize_satellite,
            "anaglyph 3D": colorize_anaglyph_3d
        }

        raw_output = domain_map[domain](input_image)

        if artistic_effect > 0:
            output_image = apply_artistic_filter(raw_output, artistic_effect)
        else:
            output_image = raw_output

    st.header("Results")

    if input_image is not None:
        tab1, tab2 = st.tabs(["Input Image", "Colorized Output"])

        with tab1:
            st.image(input_image, caption="Original Image", use_column_width=True)

        with tab2:
            if output_image is not None:
                st.image(output_image, caption=f"Colorized using {domain} domain", use_column_width=True)

                img_pil = Image.fromarray(output_image)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="Download Colorized Image",
                    data=byte_im,
                    file_name="colorized_output.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.info("Process your image to see the colorized result here.")
    else:
        st.info("Please upload an image to get started.")

    if input_image is not None and output_image is not None:
        st.subheader("Side-by-Side Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption="Original", use_column_width=True)
        with col2:
            st.image(output_image, caption="Colorized", use_column_width=True)

    st.markdown("---")

    with st.expander("About This App"):
        st.write("""
        This cross-domain image colorization tool can convert images across multiple domains:
        - **Sketch (black and white)**: Converts sketches to clean black and white images
        - **Infrared**: Applies thermal-style colorization to grayscale images
        - **Sepia tone**: Applies a nostalgic sepia tone effect
        - **Cyanotype**: Creates a blue-toned print similar to classic blueprints
        - **Satellite imagery**: Applies natural geography-based coloring
        - **Anaglyph 3D**: Creates a red-cyan 3D effect

        Adjust the artistic effect slider to enhance the visual appeal of the colorized output.
        """)

    with st.expander("How does it work?"):
        st.write("""
        #### Colorization Algorithms

        - **Sketch (black and white)**: Uses adaptive thresholding and edge detection to clean up sketches and convert them to pure black and white.
        - **Infrared**: Maps grayscale intensity to a temperature-like color gradient (blue for cold, red for hot).
        - **Sepia tone**: Applies a matrix transformation for the classic brownish-yellow aged photo effect.
        - **Cyanotype**: Creates the classic blue print effect through channel manipulation.
        - **Satellite imagery**: Uses intensity-based color mapping for terrain-like visualization.
        - **Anaglyph 3D**: Creates offset red and cyan channels to produce a stereoscopic 3D effect.

        The artistic filter enhances saturation and contrast to produce more visually appealing results.
        """)

if __name__ == "__main__":
    main()