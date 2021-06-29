## Project 3 – Disease Detection_COVID19
## Introduction:

In this project, the Operations department is part of a hospital that is taking a deep learning approach to look for patterns and methods to speed up the process of testing and diagnostics. As everyone should be familiar by now, in the beginning of the year 2020, there was the Coronavirus (Covid-19) outbreak and in this project, the goal is to tackle the challenge of developing an automated system that is able process, detect and classify these chest diseases. The drive here is to save on costs and time for detecting these chest diseases. It is the hope that the model is able to classify these diseases accurately (as well as possible) in less than 1 minute. This dataset consists of X-ray(PA-CXR) images of COVID-19, bacterial and viral pneumonia patients and normal people, where more information can be found in the "Dataset" section below. 

##### Below shows an example of a Chest CT images of a 29-year-old man with fever for 6 days:

<img src="Description Images/Coronavirus_COVID_19_CT_Scans_from_China_pub_in_Radiology.png" width="450">

Image Ref -> https://www.itnonline.com/content/ct-provides-best-diagnosis-novel-coronavirus-covid-19

It would take experience Medical Professionals several minutes or more to confirm or at least partially confirm the results of such a scan. The limiting operational factor here is that this has to be done on a case-by-case basis, essentially forming a bottleneck in the overall diagnosis process. It is the hope of a Data Scientist to build a model that can speed this part of the process up so that the conclusion can be reach at a faster rate. 

### Breakdown of the Showcase Files found in this Github Repository (seen above):
1. __Disease Detection_COVID19_Notebook1.ipynb__ - Shows my complete step-by-step approach to reaching the initial model. This will include all the theory behind ResNet-50 models. [Link](https://nbviewer.jupyter.org/github/ylee9107/DataScience_Projects/blob/main/Project%203%20-%20Disease%20Detection_COVID19/Disease%20Detection_COVID19_Notebook1.ipynb)
2. __Disease Detection_COVID19_Notebook2.ipynb__ - Shows my complete approach to reaching the final model where it has been to tuned to reach a higher accuracy and therefore better predictive power. [Link](https://nbviewer.jupyter.org/github/ylee9107/DataScience_Projects/blob/main/Project%203%20-%20Disease%20Detection_COVID19/Disease%20Detection_COVID19_Notebook2.ipynb)
3. __Disease Detection_COVID19_Main.ipynb__ - Shows the “Streamlined” version where it removes all the experiments from the notebook mentioned above, and focuses on loading in the dataset and trained ResNet-50 model to output the final results. This can be considered closer to the “Deployable” version of this project. [Link](https://nbviewer.jupyter.org/github/ylee9107/DataScience_Projects/blob/main/Project%203%20-%20Disease%20Detection_COVID19/Disease%20Detection_COVID19_Main.ipynb)


## Breakdown of this Project:
1. Loading in the Dataset.
2. Exploratory Data Analysis (Visualise the data).
3. Examining the Outliers.
4. Dataset preparation (Data cleaning, training and testing splits)
5. Building the CNN Model (ResNet)
6. Training the Model.
7. Evaluating the Model.
8. Prediction implementation.


## Dataset:

Link: https://www.kaggle.com/unaissait/curated-chest-xray-image-dataset-for-covid19

As quoted from the link, the description is:

This is a combined curated dataset of COVID-19 Chest X-ray images obtained by collating 15 publicly available datasets as listed under the references section. The present dataset 
contains:
- 1281 COVID-19 X-Rays.
- 3270 Normal X-Rays.
- 1656 viral-pneumonia X-Rays.
- 3001 bacterial-pneumonia X-Rays.

## Requirements:
- Numpy
- Pandas
- Seaborn
- Matplotlib
- scikit-learn (sklearn)
- os
- timeit
- OpenCV2  (cv2)
- glob
- tensorflow (or Keras)

## Model Output Predictions and Accuracy:
The final model was able to achieve __~88% accuracy__ in detecting the diseases.

<img src="Description Images/ResNet_FINAL_tune.PNG" width="450">

The diagram below shows the output predictions made on the testing images and it outputs what the model believes to the correct diagnosis.

<img src="Description Images/Model_Output_results.PNG" width="450">
 
## Summary:
For this project, I was able to leverage my understanding in Convolutional Neural Networks for image classification tasks, where in this case, it is the ResNet-50 model built from scratch and apply it for this problem. I was also able to improve on the initial model with better hyperparameter selection and some model architecture changes that enables this model to reach ~88% accuracy from ~75%. A further expansion to this project would be to utilise my knowledge of image segmentation to build a model that will highlight the problem areas on the X-rays scans. This in theory would help doctors identify and verify the problem areas faster and would be able to proceed to the treatment stage. 
