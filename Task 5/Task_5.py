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

# Improved colorization algorithms
def colorize_sketch(img):
    """
    Enhanced sketch colorization that works better with real photos
    """
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Invert and threshold if needed
    if np.mean(gray) > 128:  # Light background
        # Detect edges for better sketch detection
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        binary = dilated
    else:  # Dark background or already binary
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    # Create a color map
    color_map = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    
    # Apply distance transform for more natural coloring
    dist = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Generate diverse color regions
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if binary[i, j] > 0:  # If it's a sketch line
                color_map[i, j] = [0, 0, 0]  # Keep lines black
            else:
                # Use perlin-like noise for more organic coloring
                x_comp = np.sin(i / 30) * np.cos(j / 25) * 0.5 + 0.5
                y_comp = np.cos(i / 40) * np.sin(j / 15) * 0.5 + 0.5
                
                # Mix with distance for better region separation
                d = dist_norm[i, j]
                
                # Create pastel colors
                r = int(180 + 70 * np.sin(d * 5 + x_comp * 10))
                g = int(180 + 70 * np.sin(d * 8 + y_comp * 6))
                b = int(180 + 70 * np.cos(d * 6 + (x_comp + y_comp) * 3))
                
                color_map[i, j] = [r, g, b]
    
    # Smooth the color map
    color_map = cv2.GaussianBlur(color_map, (9, 9), 0)
    
    # Overlay the original lines
    result = color_map.copy()
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if binary[i, j] > 0:
                result[i, j] = [0, 0, 0]  # Keep lines black
                
    return result

def colorize_sketch_bw(img):
    """
    Sketch to black and white with improved line detection
    """
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Apply adaptive thresholding for better line detection
    if np.mean(gray) > 128:  # Light background
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        binary = dilated
    else:  # Dark background
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Invert if needed
    if np.mean(binary) < 128:
        binary = 255 - binary
    
    # Create a clean black and white image
    result = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if binary[i, j] > 128:
                result[i, j] = [255, 255, 255]  # White
            else:
                result[i, j] = [0, 0, 0]  # Black
                
    return result

def colorize_sketch_colored(img):
    """
    Fixed function for sketch to colored conversion
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY_INV, 9, 9)
    
    # Create markers for watershed
    _, markers = cv2.connectedComponents(255 - edges)
    
    # Apply watershed segmentation
    markers = watershed(-cv2.GaussianBlur(gray, (7, 7), 0), markers, mask=255 - edges)
    
    # Create a color map with distinct colors for each region
    output = np.zeros_like(img)
    num_regions = np.max(markers) + 1
    
    # Create unique colors for each region
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(100, 220, size=(num_regions, 3), dtype=np.uint8)
    
    # Assign colors to regions
    for i in range(1, num_regions):  # Skip background (0)
        mask = (markers == i)
        region_size = np.sum(mask)
        
        if region_size > 50:  # Skip very small regions
            # Get region position for color selection
            y_idx, x_idx = np.where(mask)
            center_y, center_x = np.mean(y_idx), np.mean(x_idx)
            
            # Create position-based color (for neighboring regions to have different colors)
            hue = (center_x / img.shape[1] + center_y / img.shape[0]) % 1.0
            
            # Convert HSV to RGB for more pleasing colors
            h = hue * 360  # hue: 0-360
            s = 0.7        # saturation: 0-1
            v = 0.9        # value: 0-1
            
            # HSV to RGB conversion
            c = v * s
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = v - c
            
            if h < 60:
                r, g, b = c, x, 0
            elif h < 120:
                r, g, b = x, c, 0
            elif h < 180:
                r, g, b = 0, c, x
            elif h < 240:
                r, g, b = 0, x, c
            elif h < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            rgb = np.array([int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)])
            output[mask] = rgb
    
    # Overlay the original edges
    output[edges > 0] = [0, 0, 0]
    
    return output

def colorize_infrared(img):
    """
    Infrared colorization with improved mapping
    """
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Create a simple colorization based on intensity
    result = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    
    # Color mapping (common for infrared visualization)
    # Higher temp (brighter) -> red/yellow
    # Lower temp (darker) -> blue/purple
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            intensity = gray[i, j]
            
            if intensity < 64:  # Very cold
                result[i, j] = [0, 0, intensity * 4]
            elif intensity < 128:  # Cold
                result[i, j] = [0, (intensity - 64) * 4, 255]
            elif intensity < 192:  # Warm
                result[i, j] = [(intensity - 128) * 4, 255, 255 - (intensity - 128) * 4]
            else:  # Hot
                result[i, j] = [255, 255 - (intensity - 192) * 4, 0]
    
    return result

def colorize_infrared_colored(img):
    """
    Fixed function for infrared to colored conversion
    """
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Normalize the image
    norm_img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply edge-aware filtering to create smooth regions
    bilateral = cv2.bilateralFilter(norm_img, 9, 75, 75)
    
    # Create a natural color mapping
    result = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    
    # Natural color mapping based on thermal imaging principles
    # Cooler areas (darker) -> Green/Blue (vegetation, water)
    # Warmer areas (brighter) -> Brown/Tan (earth, buildings)
    # Hottest areas -> Lighter colors (sky, reflective surfaces)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            intensity = bilateral[i, j]
            normalized = intensity / 255.0
            
            if normalized < 0.3:  # Cold objects (vegetation typically)
                # Dark green to lighter green
                green_val = int(100 + normalized * 155)
                result[i, j] = [0, green_val, int(normalized * 100)]
            elif normalized < 0.6:  # Moderate temp (ground, structures)
                # Earth tones (browns, tans)
                brown = normalized * 2 - 0.6  # 0-0.6 range
                result[i, j] = [
                    int(140 + brown * 115),  # R
                    int(100 + brown * 80),   # G
                    int(60 + brown * 40)     # B
                ]
            else:  # Warm objects (sky, reflective surfaces)
                # Light blues to whites
                blue_val = (normalized - 0.6) * 2.5  # 0-1 range
                result[i, j] = [
                    int(200 + blue_val * 55),   # R
                    int(200 + blue_val * 55),   # G
                    int(220 + blue_val * 35)    # B
                ]
    
    # Apply some smoothing to the result
    result = cv2.GaussianBlur(result, (5, 5), 0)
    
    return result

# Apply artistic filter for more interesting results
def apply_artistic_filter(img, filter_strength=1.0):
    # Convert to float
    img_float = img.astype(np.float32) / 255.0

    # Increase saturation
    hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + 0.5 * filter_strength)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1.0)

    # Adjust contrast
    img_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img_enhanced = np.power(img_enhanced, 0.9 + 0.2 * filter_strength)
    img_enhanced = np.clip(img_enhanced, 0, 1.0)

    # Convert back to uint8
    return (img_enhanced * 255).astype(np.uint8)

def main():
    st.set_page_config(page_title="Cross-Domain Image Colorization", layout="wide")

    # Initialize the app
    st.title("Cross-Domain Image Colorization")
    st.write("Upload an image to convert it to color based on different domains.")

    # Input Settings Section
    st.header("Input Settings")

    # Image upload
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    # Domain selection
    domain = st.selectbox("Select Colorization Domain:",
                         ["sketch (black and white)", "sketch to colored", "infrared", "infrared to colored", "general"])

    # Additional parameters
    artistic_effect = st.slider("Artistic Effect Strength", 0.0, 2.0, 1.0, 0.1)

    # Process button
    process_button = st.button("Process Image", use_container_width=True)

    # Initialize images
    input_image = None
    output_image = None

    # Load the image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        input_image = np.array(image)

        # Convert RGBA to RGB if needed
        if input_image.shape[-1] == 4:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)

    # Process the image
    if input_image is not None and process_button:
        # Resize if too large
        max_dim = 800
        h, w = input_image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            input_image = cv2.resize(input_image, (new_w, new_h))

        # Apply appropriate colorization algorithm based on domain
        domain_map = {
            "sketch (black and white)": colorize_sketch_bw,
            "sketch to colored": colorize_sketch_colored,
            "infrared": colorize_infrared,
            "infrared to colored": colorize_infrared_colored,
            "general": colorize_sketch,  # Using colorize_sketch as the general method
        }

        raw_output = domain_map[domain](input_image)

        # Apply artistic filter if requested
        if artistic_effect > 0:
            output_image = apply_artistic_filter(raw_output, artistic_effect)
        else:
            output_image = raw_output

    # Display results section
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

    # Side-by-side comparison
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
        This cross-domain image colorization tool can convert images from multiple domains:
        - **Sketch (black and white)**: Converts sketches to clean black and white images
        - **Sketch to colored**: Transforms sketches into colorful illustrations
        - **Infrared**: Applies thermal-style colorization to grayscale images
        - **Infrared to colored**: Applies natural color mapping to infrared-like images
        - **General**: Applies general colorization to black and white photos

        Adjust the artistic effect slider to enhance the visual appeal of the colorized output.
        """)

    with st.expander("How does it work?"):
        st.write("""
        #### Colorization Algorithms

        - **Sketch (black and white)**: Uses adaptive thresholding and edge detection to clean up sketches and convert them to pure black and white.
        - **Sketch to colored**: Uses watershed segmentation to identify regions in the sketch and assigns colors based on spatial relationships.
        - **Infrared**: Maps grayscale intensity to a temperature-like color gradient (blue for cold, red for hot).
        - **Infrared to colored**: Uses enhanced mapping to simulate natural landscape colors based on infrared intensities.
        - **General**: Uses a combination of skin tone mapping for mid-range intensities and blue/yellow gradients for shadows and highlights.

        The artistic filter enhances saturation and contrast to produce more visually appealing results.
        """)

if __name__ == "__main__":
    main()
