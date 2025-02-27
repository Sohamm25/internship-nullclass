Today, I focused on understanding CNN architecture and how its used for image colorisation. I compared how PyTorch and TensorFlow handle CNN layers and found that
In PyTorch, you need to manually specify input and output channels for each convolutional layer as it was taught in lectrure videos of null class that project which was made in pytorch but the same i am thinking to implement this task in tensorflow.
Additionally, I reviewed how our model processes grayscale images and predicts colors using convolutional layers. I now have a better idea of how the model learns to colorize images step by step. I have made the architecture code and tomoworow i will complete the task
day 3 - 
so day 3 lets focus on completion of task 1 which i have been planning for past 2 days 
1st - importing libraries 
2nd-  and then setting up data - using cifar dataset and as usual normise it by dividing by 255, then convert rgb to grayscale 
3rd - Builiding arch- where i got input image in first layer and then output image layer also included in architectre 
as i designed the architecture on day 2 it was not perfect but with trial and error and using right activation functions it works well now 
4th - next the model training set batch size and epochs and set loss as a metric to calculate how well our model works 
5th -wrote a simple code to test my model on 5 images only and it worked well
6th - last step i have used
from google.colab import files 
now i can upload grayscale image and then get its colorised version 
also the uploaded image can be of any dimension so in code i wrote img_resized = img.resize((32, 32)) 
also the model predicts in batches even if it’s just one image so i wrote this- img_array_expanded = np.expand_dims(img_array, axis=0) 
and then grayscale conversion code and then using our model and then some final visualisation code.
here is the github link of task 1 project.
https://github.com/Sohamm25/Internship---NullClass/blob/main/task1.ipynb
