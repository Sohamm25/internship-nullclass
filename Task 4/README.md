# Grayscale Colorizer

This project is an interactive web application that allows users to colorize grayscale images by defining specific regions and assigning colors to them. Users can select areas of interest in the image using coordinate-based region selection and apply custom colors to guide the colorization process.

## Features

- Upload grayscale images for colorization
- Define custom regions using X, Y coordinates with width and height parameters
- Assign specific colors to each defined region
- Preview original and colorized images side-by-side
- Download the resulting colorized image
- Responsive interface with helpful tips

## How to Run the Project

1. Make sure you have Python installed on your system
2. Install the required dependencies:
   ```
   pip install streamlit tensorflow numpy opencv-python pillow matplotlib
   ```
3. Save the provided code as `app.py`
4. Run the application:
   ```
   streamlit run app.py
   ```
5. Access the web interface through your browser (default: http://localhost:8501)

## How It Works

The application colorizes grayscale photos based on user-defined circumstances by allowing users to:

1. Upload a grayscale image
2. Select specific regions in the image using X, Y coordinates and dimensions
3. Assign custom colors to each region
4. Apply the colorization process, which blends the user-defined colors with the grayscale values

The region selection system enables precise control over which parts of the image receive specific colors. For example, you can define a region for the sky and make it blue, another region for grass and make it green, and so on.

## Model Architecture

The application uses a convolutional neural network based on TensorFlow that:
- Takes grayscale images as input
- Processes them through multiple convolutional layers
- Combines the neural network output with user-defined color constraints
- Generates colorized RGB images as output

## Tips for Best Results

- Use smaller images for faster processing
- Be patient during the colorization process
- Define regions accurately for better color application
- Experiment with different color combinations for optimal results
