## Note:
1 - The model performs best when the input images belong to the same categories as the CIFAR-10 dataset.  
2 - Since this is an image colorization task, traditional classification metrics like accuracy, precision, and recall are not suitable. Instead, loss functions such as Mean Squared Error (MSE) or perceptual loss are more appropriate for evaluating model performance, as they measure how close the predicted colors are to the ground truth. Observable colorization results are the key evaluation criteria rather than classification-based metrics

## Steps to Run

1. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
2. Open the Jupyter Notebook file.  
3. Run all the cells sequentially.  
4. In the **fifth cell**, you can check the model's performance on the dataset.  
5. In the **sixth cell**, you can upload images for testing.  

### CIFAR-10 Categories:
- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  
