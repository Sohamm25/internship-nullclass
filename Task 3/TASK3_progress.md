Today is day 10, I started Task 3: Conditional Image Colorization, where the goal is to allow users to colorize grayscale images based on specific conditions like making the sky blue or the grass green. 
Since user interaction is key, I decided to use a segmentation-based approach where users can select regions and assign colors, 
 After some research on methods like DeOldify, ChromaGAN, and Conditional Image Colorization (CIC), I finalized a plan to implement a Streamlit-based GUI. So far, I have set up the interface where users can upload a grayscale image, select specific regions, and assign colors, and I have also implemented a basic masking system to handle these selections. 
The UI is working fine, but tomorrow, I will focus on fine-tuning the colorization process, improving the blending of applied colors, 
 and testing the model with multiple images to ensure smooth performance. 
Today is Day 11, 
 - i focused entirely on implementing the conditional image colorisation logic, 
and as yesterday I had set up the basic streamlit UI and region selection, today i focused on whether colorization process works smoothly. 
I started by writing the code to apply colors based on user inputs, ensuring that the selected regions (like sky, grass, and objects) get the right color while also trying to maintain natural look. 
 also i tried doing some experimentations with different ways to color properly so that the transition doesn’t look artificial, basically i was trying to make the model better. 
i faced some issues for applying colorisation and segmentation mask, but was cleared later and i learnt some mistakes i was doing while using these pretrained models which were creating these masks, at the end i have implemented those into my project nicely
Now nearly 60 percent of task 3 is done and i am crystal clear what i have to do on day 12 and so i am sure that task 3 to be completed tomorow if i put my whole day in this task.
