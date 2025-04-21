# Task 4 Progress Report

## Day 14

Now, Task 4 nearly looks similar to Task 3 in terms of implementation. After reading the Task 4 description, I identified two major additions or changes that I needed to make to my Task 3 code:
 
1. **Interactively control the colorization process**
2. **The model should dynamically change the colorization** 

I reviewed my Task 3 code and decided to edit it. I initially integrated a CNN architecture in Task 3 to explore the possibility of using AI for automated colorization. However, I found that allowing users to manually select regions and apply colors provided more precise and user-friendly results. That’s why I decided to remove the CNN in Task 4 to simplify the tool and focus entirely on enhancing the interactive colorization process. To finalize this decision, I took help from GPT and Claude to get ideas on how to build an effective tool.

Today, I mainly worked on interactive control for the colorization. To achieve this, I decided to add a **color intensity slider** on the sidebar to control how dark the color should be. Then, I planned to implement **dynamic colorization**, which includes adding a **live preview** of the image as the user selects regions and applies colors. My entire day was spent researching how to implement this effectively.

Overall, my plan is now clear, and I believe Task 4 will not be a major challenge. I can complete it tomorrow on Day 15 if I work the whole day and do not face any major issues. My main focus will be on adding the **live preview feature** and **color intensity adjustments**.

---

## Day 15

Today, I successfully completed Task 4. I have successfully added the **live preview** and **color intensity adjustment** features, and overall, the tool now looks very refined and user-friendly.

- Users can simply upload an image and select specific regions using **interactive X, Y coordinate selection**.
- A **live preview** appears next to the region selection, allowing users to see real-time colorization updates.
- The **color intensity slider** enables users to adjust the color's density (how dark or light it should be).
- The updated image is displayed using **st.image**, providing a **dynamic preview** as users adjust the settings.
- The **color intensity slider** is implemented using Streamlit’s `st.slider`, which returns a value (`color_strength`) that determines the applied color’s intensity.

After extensive work today, I have **successfully completed Task 4**.
