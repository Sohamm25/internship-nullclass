# Task 3 Progress

## Day 10
Today is day 10, I started Task 3: Conditional Image Colorization, where the goal is to allow users to colorize grayscale images based on specific conditions like making the sky blue or the grass green. 

Since user interaction is key, I decided to use a segmentation-based approach where users can select regions and assign colors. 

After some research on methods like DeOldify, ChromaGAN, and Conditional Image Colorization (CIC), I finalized a plan to implement a Streamlit-based GUI. So far, I have set up the interface where users can upload a grayscale image, select specific regions, and assign colors, and I have also implemented a basic masking system to handle these selections. 

The UI is working fine, but tomorrow, I will focus on fine-tuning the colorization process, improving the blending of applied colors, and testing the model with multiple images to ensure smooth performance. 

## Day 11
Today is Day 11, I focused entirely on implementing the conditional image colorisation logic, and as yesterday I had set up the basic streamlit UI and region selection, today I focused on whether colorization process works smoothly. 

I started by writing the code to apply colors based on user inputs, ensuring that the selected regions (like sky, grass, and objects) get the right color while also trying to maintain natural look. 

Also, I tried doing some experimentations with different ways to color properly so that the transition doesn’t look artificial, basically I was trying to make the model better. 

I faced some issues for applying colorisation and segmentation mask, but was cleared later and I learnt some mistakes I was doing while using these pretrained models which were creating these masks. At the end, I have implemented those into my project nicely.

Now nearly 60 percent of Task 3 is done, and I am crystal clear what I have to do on Day 12, so I am sure that Task 3 will be completed tomorrow if I put my whole day in this task.

## Day 12
**Off day**

## Day 13
Today is day 13. Over the last two days, I worked extensively on Task 3 and successfully completed it. I learned several new concepts related to deep learning, image processing, and interactive UI development using Streamlit. These two days have helped me reach new heights and increased my confidence in making projects. 

So these two days I worked on basically everything because at one point it was a mess with my previous implementation on Task 3, so I started over. I used Streamlit, then built a CNN encoder architecture, Streamlit was used for uploading, and then processed images by converting them between RGB and LAB color spaces, applying user-defined color constraints to specific regions. 

For doing this, I used Matplotlib and now the region selection has become very clear and easily we can select the region we want by pasting the right x, y coordinates. Also, width and height can be adjusted, then color filling is done by mapping the grayscale intensity to the user-selected color. 

At last, you can see a **Download Colorized Image** button and also a **Model Summary** button. After many implementations, the project is finally done. 

## Note
- Try pasting low-resolution images. If images in sizes of MBs are uploaded, this can result in **15 minutes to many hours** of training and processing time. So it's better to paste a **low-resolution image** or a **compressed image**.
- In Streamlit, paste the values **one by one slowly** as simultaneous actions or quick filling might not work in Streamlit.

## GitHub Repository
[Task 3 - Conditional Image Colorization](https://github.com/Sohamm25/internship-nullclass/tree/main/Task%203)

Uploaded some screenshots as well of the project.
