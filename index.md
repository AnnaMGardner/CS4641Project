Anna Gardner (Report and Stroke Predictor), Emma Long (K-Means), Zhenming Liu (Data Preprocessing), Yawen Tan (GMM)

![stroke pic](Stroke_Web.png)

## Infographic
![Infographic](Infographic.png)

## Discover your risk of stroke
{% include stroke_input.html %}

{% include 3d_brain.html %}

This prediction is based on a supervised machine learning model trained on data from the internet and should not replace medical advice from your doctor. 

## Introduction 
Strokes are one of the most common diseases. They affect the arteries within and leading to the brain. Globally, strokes are the second leading cause of death, accounting for approximately 11% of all deaths according to the World Health Organization (WHO). There are many factors that can be used to predict a patient's risk of stroke, including high blood pressure, smoking, diabetes, high cholesterol levels, heavy drinking, high salt and fat diets, and lack of exercise. Most importantly, older people are more likely to suffer from a stroke than younger people. In addition, those who have already had a stroke are at greater risk of experiencing another. Therefore, our team aims to predict whether a patient has a high possibility to get a stroke or not based on a robust dataset. Our results can also remind those who have high-risk health measurements to change their lifestyles to avoid stroke.

## Methodology 

### Original dataset

| Total Number of Patient | Total Number of Features  | Stroke or Not? |
| ----------------------- | ------------------------- | -------------- |
|           5110          |              11            |       Y/N      |

| Patient ID | Gender | Age | Hypertension | Heart Disease | Ever Married | Work Type | Residence Type | Average Glucose Level | BMI | Smoking Status |
| ---------- | ------ | --- | ------------ | ------------- | ------------ | --------- | -------------- | --------------------- | --- | -------------- |
|  67-72940  |   F/M  | 0-82|      Y/N     |      Y/N      |     Y/N      |     4     |  Urban/Rural   |         55-271        |10-97|        4       |

## Data Preprocessing
***guys i think we should have normalized our data***
The stroke prediction dataset [1] will be used in this project. There are a total of 5110 row (number of samples) and 12 columns with 11 features and one target column. The feature columns include physiological information believed to be relative to the chance of getting a stroke. The feature column contains both string and an integer value. We implemented label coding to convert any string value to an integer value for better interpretation of the dataset. The target column is a 1-D array of boolean values indicating whether stroke risk is identified.  
Given that this dataset has only 11 features, it is not necessary to perform any dimensionality reduction for clustering analysis or our supervised learning approach. However, there were some missing data points in the BMI feature. These missing datapoints were replaced by the mean BMI of our dataset so as to minimally impact our outcomes. Additionally, the patient ID value is not relevant to stroke likelihood and was removed for our data analysis. Following all of this, we normalized our data using a standard scalar so that the encoded values of our data that was strings would not have a disproportionate impact on our results. This was all of the preprocessing done for our first round of unsupervised analysis. 
In order to improve visualization ability of our cluster analysis for KMeans and Gaussian Mixture Modeling, however, we also implemented t-distributed stochastic neighbor embedding (T-SNE) to reduce the dimensionality of our features to 2 and 3 features. 
***include pictures***
We also performed PCA in order to reduce the dimensionality to 2 and 3 dimensional spaces. We then performed a clustering analysis on these reduced datapoints as well.
***include pictures***
For the supervised learning portion of our project, a major issue in the given dataset is that the raw data is unbalanced. 249 data points identify the chance of stroke, and 4821 data points have no stroke given that stroke likelihood in the average patient is very low. In order to mitigate issues that arise from only 5% of our datapoints being for a patient who suffered from a stroke, we preprocessed the dataset using the synthetic minority oversampling technique (SMOTE) [6]. This increased the amount of datapoints that indicate stroke to 50%.
***include pictures***
The processed data was split into two segments, with 80%  for training and the remaining for testing.

## Methods
We analyzed the preprocessed dataset using two unsupervised clustering analysis approaches for expectation maximization. First we clustered using K-Means, and then with Gaussian Mixture Modeling (GMM).

## KMeans Clustering
We conducted the K-Means algorithm on both the original dataset and the dataset after T-SNE and PCA. The most important aspect of using this algorithm was to determine if the resulting clusters were useful for classifying specific attributes, in this case stroke risk, for specific groups of people.
The elbow method was used to determine the optimal number of clusters for the K-Means algorithm, which estimates the improvement for the addition of each cluster. Then we ran the K-Means algorithm for each of these optimal number of clusters.

## K-Means and GMM Results
Our data was preprocessed with 10 different combinations:
### Unbalanced data (with label encoding, filled in missing data, and dropped patient id)
Elbow Method:
![Unbalanced Data](images/unbalancedDataElbow.jpg)
Optimal Clusters = 5

K-Means:

Cluster Evaluation:

### Balanced data (with label encoding, filled in missing data, and dropped patient id)
Elbow Method:
<img src="images/balancedDataElbow.jpg" width="50"/>
Optimal Clusters = 5

K-Means:

Cluster Evaluation:

### 2D TSNE unbalanced data
Elbow Method:
![2d TSNE Unbalanced](images/2dTSNEUnbalancedElbow.jpg){width=20%}
Optimal Clusters = 5

K-Means:

Cluster Evaluation:

### 2D TSNE balanced data
Elbow Method:
![2d TSNE Balanced](images/2dTSNEBalancedElbow.jpg){width=20%}
Optimal Clusters = 5

K-Means:

Cluster Evaluation:

### 3D TSNE unbalanced data
Elbow Method:
![3d TSNE Unbalanced](images/3dTSNEUnbalancedElbow.jpg){width=20%}
Optimal Clusters = 7

K-Means:

Cluster Evaluation:

### 3D TSNE balanced data
Elbow Method:
![3d TSNE Balanced](images/2dTSNEBalancedElbow.jpg){width=20%}
Optimal Clusters = 7

K-Means:

Cluster Evaluation:

### 2D PCA unbalanced data
Elbow Method:
![2d PCA Unbalanced](images/2dPCAUnbalancedElbow.jpg){width=20%}
Optimal Clusters = 5

K-Means:

Cluster Evaluation:

### 2D PCA balanced data
Elbow Method:
![2d PCA Balanced](images/2dPCABalancedElbow.jpg){width=20%}
Optimal Clusters = 5

K-Means:

Cluster Evaluation:

### 3D PCA unbalanced data
Elbow Method:
![3d PCA Unbalanced](images/3dPCAUnbalancedElbow.jpg){width=20%}
Optimal Clusters = 5

K-Means:

Cluster Evaluation:

### 3D PCA balanced data
Elbow Method:
![3d PCA Balanced](images/3dPCABalancedElbow.jpg){width=20%}
Optimal Clusters = 7

K-Means:

Cluster Evaluation:



## GMM Clustering





## Supervised Learning
Common supervised algorithms used for stroke prediction include Decision Tree, Voting Classifier[2], and Random Forecast, Logistic Regression [3]. Throughout this project, we will construct alternative models based on the above algorithms and compare the accuracy and precision of each method. We will also measure and visualize the statistical relationships between each standalone feature and stroke likelihood.

## Results
The expected outcome of our dataset stochastic neighbor embedding is that there will exist clusters of similar patient datapoints. These clusters represent patients who have similar health feature values. Given these similarities, one or more of these clusters may represent patients with high risk of stroke. The expected relationship between health features and stroke likelihood is expected to reflect known stroke risk factors including age, diabetes, and hypertension [5].
Additionally, the expected outcome of a supervised algorithm for stroke predicion is a binary classification and prediction of the data point's stroke value. Given the relatively small size of the dataset and the need for synthetic minority data creation, we aim for an accuracy of 80% for our test data. 

## Discussion
Reaching a prediction accuracy of over 80% for stroke risk would mean that from a simple set of health measurements this model can identify a person at high risk for stroke. This model would then be able to identify which individuals should take preventative measures for strokes. 
Additionally, with a trained supervised model of high accuracy stroke prediction, we aim to create a stroke risk calculator which can non-medically predict the likelikood of patient stroke. 

## References
[1]“Stroke prediction dataset,” [Online]. Available: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset.  

[2] Tahia Tazin, Md Nur Alam, Nahian Nakiba Dola, Mohammad Sajibul Bari, Sami Bourouis, Mohammad Monirujjaman Khan, "Stroke Disease Detection and Prediction Using Robust Learning Approaches", Journal of Healthcare Engineering, vol. 2021, Article ID 7633381, 12 pages, 2021. https://doi.org/10.1155/2021/7633381.  

[3] JoonNyung Heo, Jihoon Yoon, Hyungjong Park, Young Kim, Hyo Suk Nam, Ji Hoe Heo. "Machine Learning–Based Model for Prediction of Outcomes in Acute Stroke". Stroke. 50. 1263–1265, 2019 http://doi.org/10.1161/STROKEAHA.118.024293.  

[4] Yew, Kenneth S, and Eric Cheng. “Acute stroke diagnosis.” American family physician vol. 80,1 (2009): 33-40. http://www.ncbi.nlm.nih.gov/pmc/articles/pmc2722757/  

[5] Boehme, Amelia K et al. “Stroke Risk Factors, Genetics, and Prevention.” Circulation research vol. 120,3 (2017): 472-495. doi:10.1161/CIRCRESAHA.116.308398

[6] Chawla, Nitesh & Bowyer, Kevin & Hall, Lawrence & Kegelmeyer, W.. (2002). SMOTE: Synthetic Minority Over-sampling Technique. J. Artif. Intell. Res. (JAIR). 16. 321-357. 10.1613/jair.953. 
