import streamlit as st 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image 
import numpy as np
from skimage import color 
from tensorflow.keras.models import load_model
from skimage.transform import resize

@st.cache_resource
def load_segmentation_model():
    model = load_model("my_unet_model.h5")
    return model

def get_segmentation_mask(image, model):
    image_resized = image.resize((128, 128))  # Match model input
    image_array = np.array(image_resized) / 255.0
    input_tensor = np.expand_dims(image_array, axis=0)  # Shape: (1, 128, 128, 3)
    prediction = model.predict(input_tensor)[0]

    if prediction.shape[-1] > 1:
        prediction = np.argmax(prediction, axis=-1)
    else:
        prediction = (prediction.squeeze() > 0.5).astype(np.uint8)

    mask = Image.fromarray((prediction * 255).astype(np.uint8)).resize(image.size)
    mask_np = np.array(mask) // 255
    return mask_np, mask_np

class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.encoder(x)
        ab_channels = self.decoder(features)
        return ab_channels

@st.cache_resource
def load_colorization_model():
    model = ColorizationNet()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

def colorize_grayscale_image(image, mask, model, device, region='foreground'):
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    lab_image = color.rgb2lab(image_np)
    L_channel = lab_image[:, :, 0]
    L_norm = L_channel / 100.0
    L_tensor = torch.tensor(L_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        ab_output = model(L_tensor)

    ab_output = ab_output.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    if ab_output.shape[:2] != L_channel.shape:
        ab_output = resize(
            ab_output,
            (L_channel.shape[0], L_channel.shape[1]),
            mode='reflect',
            anti_aliasing=True,
            preserve_range=True
        ) * 128

    colorized_lab = np.zeros((L_channel.shape[0], L_channel.shape[1], 3))
    colorized_lab[:, :, 0] = L_channel
    colorized_lab[:, :, 1:] = ab_output
    colorized_rgb = color.lab2rgb(colorized_lab)

    mask_bool = mask.astype(bool)
    if region == 'foreground':
        blended = image_np.copy()
        blended[mask_bool] = (colorized_rgb[mask_bool] * 255).astype(np.uint8)
    elif region == 'background':
        blended = image_np.copy()
        blended[~mask_bool] = (colorized_rgb[~mask_bool] * 255).astype(np.uint8)
    else:
        blended = (colorized_rgb * 255).astype(np.uint8)
    return blended, colorized_rgb

def main():
    st.title("Selective Image Colorization with Semantic Segmentation")
    st.write("Upload an image and choose which region to colorise.")
    
    uploaded_file = st.file_uploader("Upload an image (jpg or png)", type=["jpg", "jpeg", "png"])
    region_choice = st.radio("Select region to colorize", ("Foreground", "Background", "Entire Image"))
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original image", use_column_width=True)

        segmentation_model = load_segmentation_model()
        colorization_model, device = load_colorization_model()

        st.write("Generating segmentation mask...")
        mask, _ = get_segmentation_mask(image, segmentation_model)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        st.image(mask_image, caption="Segmentation Mask (White = Foreground)", use_column_width=True, clamp=True)

        st.write("Colorizing image...")
        region_key = region_choice.lower().split()[0]
        result, _ = colorize_grayscale_image(image, mask, colorization_model, device, region=region_key)
        result_image = Image.fromarray(result)
        st.image(result_image, caption="Result: Selected region colorized image", use_column_width=True)

        st.download_button("Download Segmentation Mask", mask_image.tobytes(), file_name="segmentation_mask.jpg", mime="image/jpeg")
        st.download_button("Download Colorized Image", result_image.tobytes(), file_name="colorized_image.jpg", mime="image/jpeg")

if __name__ == "__main__":
    main()
