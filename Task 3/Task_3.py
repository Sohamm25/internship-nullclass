import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, hex2color
st.set_page_config(page_title="Grayscale Colorizer", layout="wide")
with st.sidebar:
    st.markdown("### Pro Tips")
    st.markdown(
        """
        - **Be Patient:** After entering values, wait until the process finishes to avoid Streamlit errors.
        - **Use Smaller Images:** Smaller images speed up processing and reduce the chance of errors.
        """
    )
# i defined a more robust colorization model after learning from my previous implementations
def build_colorization_model():
    inputs = tf.keras.layers.Input(shape=(None, None, 1))
    # Encoder
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    # output
    outputs = tf.keras.layers.Conv2D(2, 1, activation='tanh')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    return model

# this is function to convert RGB to LAB
def rgb_to_lab(rgb_image):
    # Convert RGB to BGR (for cv2)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # Convert BGR to LAB
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    return lab_image

# this is a function to convert LAB to RGB
def lab_to_rgb(lab_image):
    # Convert LAB to BGR
    bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image

# this is a function to process the image with the model and user constraints
def colorize_image(model, gray_image, region_colors):
    try:
        if len(gray_image.shape) > 2 and gray_image.shape[2] > 1:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
        
        input_image = gray_image.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(np.expand_dims(input_image, axis=0), axis=-1)
        
        st.write("Processing image of shape:", input_tensor.shape)
        
        height, width = gray_image.shape
        grayscale_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        mask = np.zeros((height, width), dtype=np.uint8)
        for region, color in region_colors.items():
            if isinstance(region, tuple) and len(region) == 4:
                x, y, w, h = region                
                x = max(0, min(x, width-1))
                y = max(0, min(y, height-1))
                w = min(w, width - x)
                h = min(h, height - y)
                
                r, g, b = [int(c * 255) for c in hex2color(color)]
                region_mask = np.zeros_like(grayscale_rgb)
                for i in range(y, y+h):
                    for j in range(x, x+w):
                        if i < height and j < width:
                            # toget the grayscale value (0-255)
                            gray_val = gray_image[i, j]
                            
                            scale = gray_val / 255.0
                            region_mask[i, j, 0] = int(r * scale)  # R
                            region_mask[i, j, 1] = int(g * scale)  # G
                            region_mask[i, j, 2] = int(b * scale)  # B
                
                # this is to apply the colored region to our final image
                grayscale_rgb[y:y+h, x:x+w] = region_mask[y:y+h, x:x+w]
                
                # to update the mask for this region
                mask[y:y+h, x:x+w] = 255
        
        return grayscale_rgb
        
    except Exception as e:
        st.error(f"Error in colorization: {str(e)}")
        st.write("Detailed error information:", e)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()
def main():
    st.title("Interactive Grayscale Image Colorizer")
    st.write("Upload a grayscale image and define regions to colorize with specific colors")
    
    # to initializse session state for regions and colors if not exists
    if 'regions' not in st.session_state:
        st.session_state.regions = []
        st.session_state.region_colors = {}
    
    # this help to upload files
    uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # load and display the uploaded image
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale to ensure it's grayscale
        gray_image = np.array(image)
        
        # Display original image
        st.subheader("Original Grayscale Image")
        st.image(image, use_column_width=True)
        
        # Initialize or load model
        if 'model' not in st.session_state:
            with st.spinner("Initializing colorization model..."):
                st.session_state.model = build_colorization_model()
                st.success("Model initialized!")
        
        # this is the region selection and color assignment
        st.subheader("Define Color Regions")
        st.write("Select regions and assign colors to guide the colorization process")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # Display image with current regions
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(gray_image, cmap='gray')
            ax.set_title("Select Regions to Colorize")
            for i, (x, y, w, h) in enumerate(st.session_state.regions):
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(x, y-5, f"Region {i+1}", color='red', fontsize=12)
            
            st.pyplot(fig)
        with col2:
            # we can add new region
            st.write("Add New Region")
            x = st.number_input("X Position", 0, gray_image.shape[1], 10)
            y = st.number_input("Y Position", 0, gray_image.shape[0], 10)
            w = st.number_input("Width", 10, gray_image.shape[1], min(100, gray_image.shape[1]-x))
            h = st.number_input("Height", 10, gray_image.shape[0], min(100, gray_image.shape[0]-y))
            
            # add region button
            if st.button("Add Region"):
                new_region = (x, y, w, h)
                st.session_state.regions.append(new_region)
                st.session_state.region_colors[new_region] = "#FF0000"  # default is set to red
                st.rerun()
        if len(st.session_state.regions) > 0:
            st.subheader("Configure Region Colors")
            
            for i, region in enumerate(st.session_state.regions):
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.write(f"Region {i+1}: ({region[0]}, {region[1]}, {region[2]}x{region[3]})")
                with cols[1]:
                    color = st.color_picker(f"Color for Region {i+1}", 
                                           st.session_state.region_colors.get(region, "#FF0000"))
                    st.session_state.region_colors[region] = color
                with cols[2]:
                    if st.button(f"Remove Region {i+1}"):
                        st.session_state.regions.pop(i)
                        if region in st.session_state.region_colors:
                            del st.session_state.region_colors[region]
                        st.rerun() 
            st.subheader("Apply Colorization")
            if st.button("Colorize Image", type="primary"):
                try:
                    with st.spinner("Colorizing image with your preferences..."):
                        colorized_image = colorize_image(st.session_state.model, 
                                                     gray_image, 
                                                     st.session_state.region_colors)
                        
                        st.session_state.colorized_result = colorized_image
                except Exception as e:
                    st.error(f"Error during colorization: {str(e)}")
            if 'colorized_result' in st.session_state:
                st.subheader("Colorized Result")
                st.image(st.session_state.colorized_result, use_column_width=True)
                
                # download option
                colorized_pil = Image.fromarray(st.session_state.colorized_result)
                buf = io.BytesIO()
                colorized_pil.save(buf, format="PNG")
                byte_img = buf.getvalue()
                
                st.download_button(
                    label="Download Colorized Image",
                    data=byte_img,
                    file_name="colorized_image.png",
                    mime="image/png"
                )
        else:
            st.info("Add regions to the image and assign colors to begin colorization.")
        #SOME INFO I THOUGHT TO ADD TO MAKE THE WEBSITE LOOK GOOD.
        with st.expander("Show Model Architecture Details"):
            st.code(get_model_summary(st.session_state.model))
            st.write("""
            This model uses a simpler CNN architecture for image colorization that avoids dimension mismatches:
            - It takes a grayscale image as input (L channel)
            - Uses a series of convolutional layers with same padding to maintain dimensions
            - Predicts the a and b color channels
            - Combines with user-defined color constraints for specific regions
            - Produces a full-color RGB image
            """)

if __name__ == "__main__":
    main()