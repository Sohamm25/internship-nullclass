# Task 1 - Basic Image Colorization

### Description:
Implement a simple model that **colorizes grayscale photos** using **basic deep learning techniques**.

### Guidelines:
- Build the model with **Python** and a deep learning framework such as **TensorFlow** or **Keras**.
- Focus on achieving **observable colorization results**.
- **GUI is not necessary**.

---

## Day 1 - Understanding the Problem Statement

Today, I focused on **checking, analyzing, and understanding** the problem statement for Task 1. The key takeaways included:

- Understanding the importance of **image colorization** in deep learning.
- Reviewing existing **research papers and previous implementations**.
- Exploring datasets that could be used for training.
- Deciding to use **TensorFlow/Keras** for implementation.

This initial research provided a solid foundation for moving forward with the project.

---

## Day 2 - Understanding CNN for Image Colorization

Today, I focused on understanding **CNN architecture** and how it's used for **image colorization**. I compared how **PyTorch** and **TensorFlow** handle CNN layers and found that:

- In **PyTorch**, you need to manually specify **input and output channels** for each convolutional layer. This was demonstrated in the lecture videos of **NullClass**, where the project was implemented in PyTorch.
- I am considering implementing the **same task in TensorFlow** to explore differences.

Additionally, I reviewed how our model processes **grayscale images** and predicts colors using convolutional layers. Now, I have a better idea of how the model learns to colorize images **step by step**. I have designed the **architecture code**, and tomorrow, I will complete the task.

---

## Day 3 - Completing Task 1

Today, my focus was on completing **Task 1**, which I had been planning for the past two days. The workflow included:

### Steps:

1. **Importing Libraries** 📌
   ```python
   import numpy as np
   import tensorflow as tf
   from tensorflow import keras
   from tensorflow.keras import layers
   ```

2. **Setting up Data** 📂
   - Using the **CIFAR dataset**.
   - Normalizing it by dividing by **255**.
   - Converting **RGB to grayscale**.

3. **Building Architecture** 🏗️
   - The first layer takes the **input image**.
   - The output layer generates the **colorized version**.
   - Trial and error were used to find the **right activation functions**.

4. **Model Training** 📊
   - Set **batch size** and **epochs**.
   - Used **loss as a metric** to evaluate model performance.

5. **Testing the Model** 🧪
   - Wrote a simple test script for **5 images only**.
   - The model worked **successfully**!

6. **Final Deployment & Image Uploading** 📸
   - Used Google Colab for image uploading:
     ```python
     from google.colab import files
     uploaded = files.upload()
     ```
   - Allowed uploading images of **any dimension**, resizing them:
     ```python
     img_resized = img.resize((32, 32))
     ```
   - Ensured **batch processing**, even for a **single image**:
     ```python
     img_array_expanded = np.expand_dims(img_array, axis=0)
     ```
   - Added final **visualization code** to display results.

---

## 🔗 GitHub Repository
Check out the full project **[here](https://github.com/Sohamm25/Internship---NullClass/blob/main/task1.ipynb)**.

---

✨ **Next Steps:** Continue refining the model and exploring improvements! 🚀
