# Daily Progress Report

## Day 16 – Off Day

*(no activities recorded)*

## Day 17

i took a off day on day 16,  
today is day 17- as i had reached the last task of the five tasks I have given I decided to revisit or go through my previous 4 tasks today and as I had learnt so many things while I was completing these four tasks so that's why I decided to check my code files for the previous tasks and I found out that there were some errors and also some fixes that I needed to do and edited every task file on my github.  

And then today I started with my task 5 and as usual I prepared the plan to complete this task so first I I decided to use my task four file for this and after some implementations i have decided to create a new file and start from scratch for this project .  

i have understood the problem statement and what I have to do so whatever techniques that were be going to be used here I have explor0de and research about them today.  

As usual, I began by researching different approaches and existing methods to handle domain-specific colorization effectively.  

I reviewed techniques like DeOldify, ChromaGAN, and domain-specific GANs to understand their potential use for this task.I explored how to structure the GUI to allow users to select an image type and preview the colorized output dynamically basically some streamlit implementations on test.py file and got idea how will i be able to build this web tool tomorow.

Tomorrow, i will finalize the approach and start implementing the basic UI and also the project..

## Day 18 (13th March)

I am submitting day 18 and 19 report at once as yesterday due to network error I was unable to upload my progression report so  

day 18 report(13th march) - i have understood the problem statement and I am thinking to first of all finalize the domains so I can see clearly I have to turn a image into infrared then also i can turn it to a sketch or colour it, etc,  

also I have was researching on how can I make all those domains happen I found out about CV2 library and also about the edge detection it can do and i can convert images from one domain to another and i guess this method is traditional way as per my research from various resources.  

So my day 18 went completely on the working of this library I have learnt two things today it was about assigning colours to black and white images for that I used canny for edge detection and Dilation (cv2.dilate) (to enhance edges) and distance transform method from the CV2 library for the transition and then generated a colour map for it,  

then for the another function that I made using this library was to convert a sketch into black and white version and here the process in short was I used adaptive threshold that is a method from this library and at the end of the day I was able to get my results working well i didntuse any train model or i didnt train any model but I learnt that we can do it this traditional way as well.

## Day 19 (14th March)

Day 19 report (14th march)-  now due to my yesterday's research I found another way to complete this project traditional without using models without even training a model so I was thinking to continue today I made another function of infrared to colour and in short i used cv2.applyColorMap this mimics the infrared heat signature effect seen in thermal images and cv2.Canny() to sharpen features before applying color. And today after implementing all the functions I tried to integrated all of them all together in the streamlit web application.  

i did some tests and i had many issues and i kept on working those issues. A lot of time was invested today for this library of CV2 as I went on to keep researching many things I had in my path,  

Finally I have made a very good looking UI and tested all my functions that I have created using this library and it was absolutely amazing to see the results work very well.U just have to upload a image then the colorising feature or sketching or turning the image into infrared was working way more fast than the one i did so overall I learnt that when I use a pretrained model or use deep learning technique it is not as fast as the CV2 library, this library is very fast for task like these and after my extensive work and many many implementation from the past 2 - 3 days I found out this as an ideal technique but the difference is not that much so over all using a pretrainend model will be best in cases for complex tasks but for smaller task where images are involved this library works well too.

## Day 20

So today is day 20 and I was trying this CV library yesterday and I have completed the task 5 using this library but I was trying to use some pretrained models for completing these tasks, now the problem is that when are u operating on operations or tasks of gen ai then computational work or overall time and use of resurces i pretty high , i am saying this from todays expeirence because i had to download 1.5 gb minimum size of models and use them to complete those tasks which was very difficult as i had decided some domains and so for each domain i had seperate seperate files in size of GBs, so today i have downloaded them all and i have tested all those model by creating some basic functions as using them was not very difficult and after extensive work and research today i had another approach nearly ready for the project but overall i am stuck on whic apporoach to take as if i go with cv2 its compatible and conventional, faster as well, but using pretraineed models take time and also high use of resoucres so i am thinking to make that decision tomorrow and finally complete all my tasks

## Day 21

so day 21 was a tough day beacuse i had to explore the area of pretrained models being used in the code which were imported from famous open source github repositories, for ex-  
1 - colorise model - https://github.com/jantic/DeOldify?tab=readme-ov-file  
2 - sketch to color - https://drive.google.com/file/d/1MwsExdh_qViygaZ8tYlPBu9R2YGl0TeQ/view?usp=drive_link  

now these models were in size of gbs and it took a lot of time for me to get used to these models usage in my project, now here using these kind of models needed some experience so i had to create a sperate files for each domain and try each and every model out, now doing so caused many many issues which i had to research everytime and sort it out, but at the end of day i had my sketch model and infrared model and colorised model working really well, now tomorrow i am thinking of completing task 5 and preparing resouces and making repository ready for final submission. i am thinking to give my complete day tomorrow so i will complete all tomorrow itself.

## Day 22

today is day 22 - i have completed my task 5 and finally all the tasks are completed, so i decided to review my each and every project and put together all the points, topics, concpets and techniques i learned , so todays day went in completion of task 5 which invovled adding or you can say testing some features for each domain i have like the color strength and then the sketch line strength, the color intensity all these things which already where the part of the libraries i was using so it was not very hard to implement and test out and then after few more implementations i have finalised and finished my task 5 and then i went on to reviewing my previous tasks code, later i focused on the final submission i needed to submit so i prepared all the important and required files and i am left with just a few things , so i am thinking to do my final submission tomorrow , so tomorrow probably might be the last day of my internship.

## Day 23

Today is day 23 - i have prepared my report and also rearranged my github repositories and with help of gpt made those markdowns file looking good, then i focused on the files whether they meet the requirements or not, and i found some issues in task 4 so i have sorted a major flaw out , and then i checked whether i have saved the model architecture of all and also i rechecked all those model architecture by loading it back again, so my complete day went with all the checking work, fixing issues, making report and preparing resources for final submission , tomorrow is my last for sure as i will be submitting it tomorrow.

## Day 24

today is day 24- i know i said today will be my final discussion but from the mail I got about the usage of pretrained model which can lead to my disqualification so I decided to create a Jupiter file and for the every model I have used in every other task I will create a model right from the scratch so todays complete day went on training these models right from the scratch I started with task number 2 in which I focus on the colourisation model and the segmentation Mask model here I face a lot of issues while creating the segmentation model like getting a proper data set and then the architecture was the hardest part also the accuracy was also an issue, then later on I focus on task number 3 and task number 4 and I kept on training the models that I used for each and every file there, I have completed 50% of the work or you can say the remaining work and let's hope tomorrow I finish everything up.

## Day 25

Today is day 25- I have been working on creating models right from scratch for each and every task and so its taking some time as finding the right data set and planning the architecture for each and every task is a challenge for me ,  
Second thing is that it is hard to integrate or import or use this models in the main python code where the graphical user interface code is also present.  
I also tried some tensor flow databases and carried out some implementations,  
Tomorrow i will try to complete the remaining work.

## Day 26

Today is day 26 and I have been training the segmentation model today although I had the accuracy of 90% but I thought that if I do some hyperparameter tuning and put some efforts I will certainly be able to get some more percentage and as my segmentation mask were not that accurate but after increasing the number of epochs and inclusion of learning rate and some changes in the architecture I got accuracy of 96% today and the results were very very good the same went for the rest of the model training as well.  
Now my main focus is to complete task number 5 as there will be various models for various domains and I have started with normal to infrared model also normal to sketch model and I have nearly completed both of them and again the architecture was a major issue in both of these models but I researched a lot and have got good accuracy on both of these models and I'm going to integrate them tomorrow.

## Days 27 & 28

day 27 28 - off days actually preparation for nqt so coudlnt work.  
Due to power issues, i couldnt upload my report at my usual time 11:30 PM.I continued working offline and -

## Days 29 & 30

so this is the daily report of day 29 and day 30 -  
i have spent these 2 days working on model creations for each task, and as 5th task is going to be a major challenge for me as i have to create models for infrared to normal, normal to infrared and then normal to sketch, now a challenge for these model creation is that i have create a different different architectures  
starting wit infrared to normal a cycleGAN-based approach might work best because infrared and normal images have structural similarities but different distributions , this was researched with the help of various resources available,  
then for normal to sketch this transformation focuses more on edge detection and stylization, so a U-Net or a lightweight CNN trained on paired data (normal images and their sketch versions) might be ideal i think.  
datasets is also a major issue as we have to find the best datasets that match because i learnt one thing , i had used oxii pet dataset and trained my model but as it was a pet dataset it only performed well on images of pet and then i took another dataset with 10 categories but that too didnt work when the image that i upload was of other category then the model performed very badly so i had to look for a dataset that had many categories  
for Infrared to Normal - Public datasets like FLIR exist, i had tried that out too  
and the biggest issue for all these days was the overfitting issue, i face it so many time and as i had to retrain the model, it took a lot of time as from the evaluation metrics i had to get idea about how my model is performing then make the change then again check whether it will not overfit , and imagine this for all the domains so it is taking alot of time for me to cover these things.

## Day 31

day 31 - today was my nqt test, but right after i finished my nqt i went home and started from where i left off, i found some datasets after doing some research and asking gpts, then using the tensorflow and torch libraries and using their datasets i tried to train the models, many datasets where unavailable or had some issues , some datasets where of 40 GB , so i had to download some datasets from kaggle and imported them to train models, finally after few hours i have completed the normal to sketch model which with the help of u net architecture was done and also it was pretty accurate and generated sketches accurately. i have started the training of infrared to normal once again as due to some consistent issues i had faced so i started all over to get idea where i am wrong,  
i found out it was the way i was taking the inputs, that technique was wrong and so i fixed it and moved on to complete the model

## Days 32–34

day 32 - 34  
Today again, as per given recommendations for creating a segmentation mask model i tried to implement it,  
Now as given i was trying to implement the loading of dataset from various datasets al together but that didn't work that well as after training the models multiple times it was just not right, so i went to stack over flow, more websites and papers for using a database in order to get idea , then i tried the coco dataset again i tried augmentation etc , then i tried another technique or a way i noticed on those sources i was referring from so i tried to used mixed datasets from voc cityscapes coco altogether and trained the model few times which raised issues of model accuracy being lower than 0.01,  
Atlast i tried fine-tuning model in all ways possible by changing the parameters on trial error basis and kept on improving the model, i then used light weight architecture and trained the model  
So lets hope i soon tackle the segmentation mask model problem i will get into it tomorrow, and the rest of task are nearly ready

## End of March 30 & 31

On march 30 and 31st, i rechecked my 1st 3rd and 4th assignments,  
I run all the codes and provided a read me fike for each task,  
Then i focused on verifying whether i have had completed the right task as given for me, also i found out that task 4 had live preview issues so i had to research and fix it,  
Then i made report again for these tasks,  
I then decided to complete task 3 in two ways one way original way of having to select a region using x and y, 2nd i went training a model only for images of nature with sky and grass in order to check whether that will be a right solution and after many implementations, i found out the best way was the previous one as it gave user more freedom

## April 2nd

Today, on April 2nd, I continued working on segmentation models and experimented with sketch and infrared-based models. I spent time practicing with various datasets, testing their suitability for different tasks. One key observation was that the behavior of datasets changed significantly depending on the categories I included.  
This variation impacted how the model learned features, and I could see noticeable differences in performance, generalization, and output quality. It made me realize how crucial dataset selection is when training deep learning models not just about quantity but also the right mix of data.  
Through this practice, I gained a better understanding of how different datasets influence model training and performance. Theres still a lot to explore, but today was a productive step in refining my approach to dataset selection and model fine-tuning.

## Interactive User-Guided Colorization (Task 4)

Today, I worked on Interactive User-Guided Colorization which is task 4 and focussed on fixing issues related to the live preview feature in the GUI.  
at start, the model was processing user inputs correctly, but the real-time preview wasn’t updating as expected. The colorization results were only reflecting after the full processing step, which wasn’t ideal for an interactive experience.  
After debugging and checking code i found that the issue was with how the GUI handled update mainly a delay in rendering due to inefficient event handlingI worked on improving the process so that color changes show up immediately when the user selects a region and chooses colors. The update made it faster, but I still need to make it smoother.  
Overall the model is working fine right now and i also worked on task 2 and taks 5 again .

## Task 5 Updates

In Task 5, i have developed a basic Opencv based solution to convert normal images into infrared-style visuals.  
i did that by converting the input image to grayscale. This helps in extracting intensity values from each pixel, which can be interpreted as a rough estimation of  temperature darker pixels representing cooler areas and brighter pixels representing hotter regions. Based on these intensity values, a color mapping was applied to create the infrared effect., low-intensity pixels were mapped to blue or purple shades indicating cooler regions, medium-intensity pixels were mapped to greenish or yellow shades, and high-intensity pixels were assigned red or orange colors (indicating hot spots).  
this was implemented using simple if-else conditions in a pixel-wise loop over the grayscale image.  
To enhance the visual quality and add a bit of realistic ness, an alternative version of the function was also created. This version used CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast in the grayscale image before applying the color mapping. As a result, the final output appeared more vivid and natural, closely resembling how thermal cameras represent varying temperature zones.  
I had to reseacrh about all this so took a time and my complete day.

## Artistic Filter Feature

yesterday I focused on adding the artistic filter feature to enhance the final output in my colorization tool. Initially, I tried just increasing the saturation using OpenCV’s HSV conversion, but the result looked too harsh. Then I attempted contrast adjustments directly on RGB, which caused some images to get overexposed. After a bit of research and testing, I found a better method first converting the image to float, adjusting saturation in HSV, then using np.power() for smooth contrast enhancement. I fixed issues like extreme brightness by adding np.clip() to limit values. Now, the feature works well and users can adjust the effect using a slider in Streamlit for a more stylized colorized image.

## Real-Time Artistic Filter Integration

Then the next day that is, today i worked on making the artistic filter feature align smoothly with the live preview system. At first, the changes weren’t reflecting in real-time, and the preview kept lagging or updating only after a few seconds. I realized this was due to the processing happening after region selection instead of instantly. so, i refactored the code to integrate the artistic filter directly into the image update function. This way, as soon as the user adjusts the slider, the preview updates immediately. After a few time of testing and tweaking the rendering flow in Streamlit, the entire setup became much more responsive and now gives a seamless and interactive experience.

## Infrared-to-Color Model Development

Today I actively worked on developing the infrared to color model. I tried different ways to map grayscale infrared images to realistic colors. Initially, I experimented with OpenCV’s built-in color maps, but the output looked too artificial or flat depending on the intensity levels in the image. thus then i shifted to using CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast before applying any color mapping, and this actually gave much better results.,  
i also learned how sensitive infrared images are to changes in brightness and contrast. Even small tweaks in the normalization or mapping logic made a noticeable difference. One of the key takeaways from today was realizing that applying color is not just about converting pixel values but about creating a visual that feels natural to the human eye.  
Overall, today was productive in understanding the complexity behind infrared image interpretation, and the trials I ran will definitely help improve the final quality of the model.

## Unified Multi-Domain Colorization Model

Today, I worked on building a more powerful colorization model that could handle multiple domains — grayscale, sketch, and infrared — using a unified architecture. I used the CIFAR-10 dataset again but focused more on improving the structure of the model. I implemented a U-Net inspired architecture with skip connections to better capture image features and preserve important details during decoding.  
One challenge I faced was preparing and combining data from different domains (grayscale, sketch-like edges, and synthetic infrared). Especially with the sketch data, I had to tweak the Canny edge detector and normalization steps to get cleaner outputs that wouldn't confuse the model. Training was smoother this time, and the model is now able to reasonably reconstruct colors from all three types of inputs.  
The model is saved and also includes an upload function to test new images from any domain. I’ll continue refining this for better quality and maybe explore other datasets next.
