import streamlit as st 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image 
import numpy as np
from skimage import color 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error

# FIRST OF ALL SEMANTIC SEGMENTATION --
@st.cache(allow_output_mutation=True)
def load_segmentation_model():
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    return model, device

def get_segmentation_mask(image, model, device):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    mask = (output_predictions != 0).astype(np.uint8)
    return mask, output_predictions

#evaluation
def evaluate_segmentation(true_mask, pred_mask):
    acc = accuracy_score(true_mask.flatten(), pred_mask.flatten())
    prec = precision_score(true_mask.flatten(), pred_mask.flatten(), average='macro')
    rec = recall_score(true_mask.flatten(), pred_mask.flatten(), average='macro')
    conf_matrix = confusion_matrix(true_mask.flatten(), pred_mask.flatten())
    return acc, prec, rec, conf_matrix

# NEXT IS COLORISATION
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        # a simple encoder - decoder architecture.
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

@st.cache(allow_output_mutation=True)
def load_colorization_model():
    model = ColorizationNet()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

def colorize_grayscale_image(image, mask, model, device, region='foreground'):
    image_np = np.array(image)
    lab_image = color.rgb2lab(image_np)
    L_channel = lab_image[:, :, 0]
    L_norm = L_channel / 100.0
    L_tensor = torch.tensor(L_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        ab_output = model(L_tensor)
    ab_output = ab_output.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 128
    if ab_output.shape[:2] != L_channel.shape:
        ab_output = np.array(Image.fromarray(ab_output.astype(np.float32)).resize(
            (L_channel.shape[1], L_channel.shape[0])
        ))
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

def evaluate_colorization(original, generated):
    original_gray = color.rgb2gray(original)
    generated_gray = color.rgb2gray(generated)
    mse = mean_squared_error(original_gray.flatten(), generated_gray.flatten())
    return mse

def main():
    st.title("Selective Image Colorization with Semantic Segmentation")
    st.write("Upload an image and choose which region to colorise.")
    uploaded_file = st.file_uploader("Upload an image rec is jpg or png", type=["jpg", "jpeg", "png"])
    region_choice = st.radio("Select region to colorize", ("Foreground", "Background", "Entire Image"))
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original image", use_column_width=True)
        segmentation_model, device = load_segmentation_model()
        colorization_model, device = load_colorization_model()
        
        st.write("Generating segmentation mask...")
        mask, pred_mask = get_segmentation_mask(image, segmentation_model, device)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        st.image(mask_image, caption="Segmentation Mask (White = Foreground)", use_column_width=True, clamp=True)
        
        # dummy true mask
        true_mask = mask.copy()
        acc, prec, rec, conf_matrix = evaluate_segmentation(true_mask, pred_mask)
        st.write(f"Segmentation evaluation - Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)
        
        st.write("Colorizing image...")
        region_key = region_choice.lower().split()[0]
        result, generated_rgb = colorize_grayscale_image(image, mask, colorization_model, device, region=region_key)
        result_image = Image.fromarray(result)
        st.image(result_image, caption="Result: Selected region colorized image", use_column_width=True)
        
        # Download buttons
        st.download_button("Download Segmentation Mask", mask_image.tobytes(), file_name="segmentation_mask.jpg", mime="image/jpeg")
        st.download_button("Download Colorized Image", result_image.tobytes(), file_name="colorized_image.jpg", mime="image/jpeg")
        
        #evaluation
        mse = evaluate_colorization(np.array(image), generated_rgb)
        st.write(f"Colorization Evaluation - MSE: {mse:.2f}")
        
if __name__ == "__main__":
    main()
