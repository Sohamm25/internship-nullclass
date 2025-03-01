Today is day 4 and after the completion of first task I moved on to the second task where in short I have to add semantic segmentation in my project. So the first thing was understanding what I have to do in my second task and I found out that first I need to add semanti segmentation model to separate regions in my previous model for colourisation which needs to be reused here. Second thing was either I have to train a model or use of Pre trained semantic segmentation model so I did some research about such models and found out about these three - 
DeepLabV3+ or Mask R-CNN
The third step will be building graphically user interface which will allow user to select the regions to colorise 
Then the fourth step I think will be to apply the colour on the selected region 
and the fifth step will be to display the result in the GUI 
So overall my plan is to make this tool which has the graphical user interface and a model which can first colourise on a grayscal image and then according to the selection done by the user it will colourise the specific regions only. 
So I have started making the project and I have decided to use a pre -trained model and then a major challenge in front of me is to integrate my previous colour risation project with this one 
 also after doing some research and having basic knowledge of the projects that have made in past I have decided to use streamlit which is easy. 
I watched few youtube videos about those pretrained models to learn about them so that on day 5 I can start the project-
https://youtu.be/tJHMcDtfdDI?feature=shared 
https://youtu.be/mgdB7WezqbU?feature=shared
 day5 -
  On Day 5 as per previous plannings from day 4 i tried to integrate it with task 1 project but on trying and implementations i think its better to create a seperate project which will be combination of both the tasks, 
also gui was necessary in task 2 thus i have decided to use streamlit as i am very experienced when it comes to streamlit projects , 
now starting with project i tried using deeplabv3+, 
 i found out this model was unavailable or some availability issues thus decided to switch to UNET - by referring some yt vids i got idea of how to use that pretrained model, 
then i have thought of the process to create a streamlit application and decided how its going to look like as planning first will be beneficial then,
at last i have written some basic code and did some research on how these pretrained models work and tomorow on day 6 i am sure i will be able to complete the task 2
day 6 -
Today is day 6 and as I planned on day 5 about completion of the task on day 6 itself was not possible it was too much of our work but I have covered 70% of task 2 project first of all streamlit implementation was done correctly ,even the segmentation mask is getting applied correctly, 
 and then I researched a lot about how can i implement the part where parts user can select to colourise a specific region. 
However, I faced some issues while integrating it completely, which caused a delay in completing the task today, so Tomorrow i will work on applying the colorization to the segmented regions and fine-tuning the process for better results. Once that is done, Task 2 will be complete.
