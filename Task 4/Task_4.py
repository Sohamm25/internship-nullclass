import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color

st.set_page_config(page_title="Interactive Grayscale Colorizer", layout="wide")

with st.sidebar:
    st.markdown("### Interactive Colorization Controls")
    st.markdown(
        """
        - **Be Patient:** After selecting regions, wait a moment for the live preview to update.
        - **Use Smaller Images:** Smaller images speed up processing and reduce the chance of errors.
        """
    )
    st.markdown("---")
    current_color = st.color_picker("Select color to apply", "#FF0000")
    st.markdown("### Color Application Settings")
    color_strength = st.slider("Color Intensity", 0.1, 1.0, 0.7, 
                             help="Adjust the intensity of applied colors")
    
    # this is the clear button code i wrote
    st.markdown("---")
    if 'regions' in st.session_state and st.button("Clear All Regions"):
        st.session_state.regions = []
        st.session_state.region_colors = {}
        st.rerun()

#to only colors specific regions
def colorize_image(gray_image, region_colors, color_strength=0.7):
    try:
        if len(gray_image.shape) > 2 and gray_image.shape[2] > 1:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
        
        height, width = gray_image.shape
        grayscale_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        colorized_image = grayscale_rgb.copy()
        
        for region, color in region_colors.items():
            if isinstance(region, tuple) and len(region) == 4:
                x, y, w, h = region                
                x = max(0, min(x, width-1))
                y = max(0, min(y, height-1))
                w = min(w, width - x)
                h = min(h, height - y)
                r, g, b = [int(c * 255) for c in hex2color(color)]
                
                for i in range(y, y+h):
                    for j in range(x, x+w):
                        if i < height and j < width:
                            gray_val = gray_image[i, j]
                        
                            scale = gray_val / 255.0
                            current_rgb = colorized_image[i, j].copy()
                            new_r = int(r * scale * color_strength + current_rgb[0] * (1 - color_strength))
                            new_g = int(g * scale * color_strength + current_rgb[1] * (1 - color_strength))
                            new_b = int(b * scale * color_strength + current_rgb[2] * (1 - color_strength))
                            
                            colorized_image[i, j] = [new_r, new_g, new_b]
        
        return colorized_image
        
    except Exception as e:
        st.error(f"Error in colorization: {str(e)}")
        st.write("Detailed error information:", e)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

def main():
    st.title("Interactive Grayscale Image Colorizer")
    st.write("Upload a grayscale image and interactively colorize regions with real-time preview")
    
    # we have to initialize session state for regions and colors if not exists
    if 'regions' not in st.session_state:
        st.session_state.regions = []
        st.session_state.region_colors = {}
    
    # to file uploader
    uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale to ensure it's grayscale
        gray_image = np.array(image)
        st.subheader("Original Grayscale Image")
        st.image(image, use_column_width=True)
        st.subheader("Interactive Region Selection")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.write("1. Select a region by specifying coordinates")
            col_a, col_b = st.columns(2)
            with col_a:
                x = st.number_input("X Position", 0, gray_image.shape[1], 10)
                y = st.number_input("Y Position", 0, gray_image.shape[0], 10)
            with col_b:
                w = st.number_input("Width", 10, gray_image.shape[1], min(100, gray_image.shape[1]-x))
                h = st.number_input("Height", 10, gray_image.shape[0], min(100, gray_image.shape[0]-y))
            if st.button("Add Region with Selected Color"):
                new_region = (x, y, w, h)
                st.session_state.regions.append(new_region)
                st.session_state.region_colors[new_region] = current_color
                st.rerun()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(gray_image, cmap='gray')
            ax.set_title("Selected Regions")
            for i, (x, y, w, h) in enumerate(st.session_state.regions):
                color = st.session_state.region_colors.get((x, y, w, h), "#FF0000")
                rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                plt.text(x, y-5, f"Region {i+1}", color=color, fontsize=12)
            
            st.pyplot(fig)
        
        with col2:
            # and this is live preview of colorization
            st.write("Live Preview")
            
            if len(st.session_state.regions) > 0:
                colorized_result = colorize_image(
                    gray_image, 
                    st.session_state.region_colors,
                    color_strength=color_strength
                )
                
                st.session_state.colorized_result = colorized_result
                
                # this to display live preview
                st.image(colorized_result, use_column_width=True)
                
                #the Download option
                colorized_pil = Image.fromarray(colorized_result)
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
                st.info("Add regions to see the live colorization preview")
    
        if len(st.session_state.regions) > 0:
            st.subheader("Manage Color Regions")
            
            for i, region in enumerate(st.session_state.regions):
                cols = st.columns([2, 1, 1])
                with cols[0]:
                    st.write(f"Region {i+1}: ({region[0]}, {region[1]}, {region[2]}x{region[3]})")
                with cols[1]:
                    update_color = st.color_picker(f"Update color for Region {i+1}", 
                                                 st.session_state.region_colors.get(region, "#FF0000"),
                                                 key=f"update_color_{i}")
                    
                    if update_color != st.session_state.region_colors.get(region, "#FF0000"):
                        st.session_state.region_colors[region] = update_color
                        st.rerun()
                with cols[2]:
                    if st.button(f"Remove Region {i+1}"):
                        st.session_state.regions.pop(i)
                        if region in st.session_state.region_colors:
                            del st.session_state.region_colors[region]
                        st.rerun()
        
        with st.expander("About This Colorization Tool"):
            st.write("""
            This interactive colorization tool:
            - Allows real-time color selection and preview
            - Lets you control color intensity with the strength slider
            - Provides immediate visual feedback when adding or updating regions
            - Preserves the luminance details of the original grayscale image
            """)
        
        with st.expander("Download Model Files"):
            #to Download model weights
            try:
                with open("model_weights.h5", "rb") as f:
                    weights_data = f.read()
                st.download_button(
                    label="Download Model Weights",
                    data=weights_data,
                    file_name="model_weights.h5",
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.write("Model weights file not found.")
            
            #to Download saved model
            try:
                with open("saved_model.h5", "rb") as f:
                    saved_model_data = f.read()
                st.download_button(
                    label="Download Saved Model",
                    data=saved_model_data,
                    file_name="saved_model.h5",
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.write("Saved model file not found.")
    
if __name__ == "__main__":
    main()
