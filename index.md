Anna Gardner (Report and Stroke Predictor), Emma Long (K-Means), Zhenming Liu (Data Preprocessing), Yawen Tan (GMM)

<img src="Stroke_Web.png" style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 100%;"/>

## Infographic
<img src="Infographic.png" style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 100%;"/>

## Discover your risk of stroke
{% include stroke_input.html %}

{% include 3d_brain.html %}

This prediction is based on a supervised machine learning model trained on data from the internet and should not replace medical advice from your doctor. 

## Introduction 
Strokes are one of the most common diseases. They affect the arteries within and leading to the brain. Globally, strokes are the second leading cause of death, accounting for approximately 11% of all deaths according to the World Health Organization (WHO). There are many factors that can be used to predict a patient's risk of stroke including high blood pressure, smoking, diabetes, high cholesterol levels, heavy drinking, high salt and fat diets, and lack of exercise. Most importantly, older people are more likely to suffer from a stroke than younger people. In addition, those who have already had a stroke are at greater risk of experiencing another. Therefore, our team aims to predict a patient's risk of stroke based on a robust dataset. We hope to create an interactive web component to display our results, and remind those who have high-risk health measurements to act preventatively and change their lifestyles to avoid stroke.

## Methodology 

### Original dataset

| Total Number of Patients | Total Number of Features  | Stroke or Not? |
| ----------------------- | ------------------------- | -------------- |
|           5110          |              11            |       Y/N      |

| Patient ID | Gender | Age | Hypertension | Heart Disease | Ever Married | Work Type | Residence Type | Average Glucose Level | BMI | Smoking Status |
| ---------- | ------ | --- | ------------ | ------------- | ------------ | --------- | -------------- | --------------------- | --- | -------------- |
|  67-72940  |   F/M  | 0-82|      Y/N     |      Y/N      |     Y/N      |     4     |  Urban/Rural   |         55-271        |10-97|        4       |

The stroke prediction dataset [1] will be used in this project. There are a total of 5110 rows (number of samples) and 12 columns with 11 features and one target column. The feature columns include physiological information believed to be relative to the chance of getting a stroke. The feature column contains integer values such as BMI and Glucose levels. It also contains string values such as Gender. It also contains boolean values such as known history of heart disease. The target value is a discrete value in which 0 corresponds to no stroke and 1 corresponds to a stroke. 

### Data Preprocessing
In order to prepare our data for both unsupervised and supervised analysis, we cleaned, standardized, reduced the dimensionality, and synthetically balanced our raw dataset. Some features in the raw data contain string values which are difficult for a machine learning algorithm to process. We converted these feature values into integer value with label encoding. For example, in the “gender” column, the “male” value is converted into 1, while the “female” value is 0.
We also observed that in our raw data had some missing values for BMI. Given that only 3.9% of this data was missing, we kept this feature and filled any missing values with the mean value of the data column. 
To better understand the features in the data after label encoding and filling in missing data, we plotted the correlation heat map shown in Figure 1. Features with very high correlation to each other and very low correlation to the target are subject to be dropped to reduce the overall dimensionality of our data. Due to the low correlation value between the “id” and the target ”stroke” we dropped this feature.

After dropping the “id” feature, we performed SMOTE to balance the data. A major issue in the given dataset is that the raw data is unbalanced. 249 data points identify the chance of stroke, and 4821 data points identify no stroke given that stroke likelihood in the average patient is very low. In order to mitigate issues that arise from only 5% of our datapoints being from a patient who suffered from a stroke, we also rebalanced the dataset using the Synthetic Minority Oversampling Technique (SMOTE) [6]. This process chooses samples with the same target value that are close to eachother in the feature space and selects new datapoints that exist on a line between them. The balanced data oversamples at the adjacent of the minority (positive) datapoints to have the same number of data points as the majority (negative) data (Figure 2 in results). We applied this method before performing PCA and T-SNE dimensionality reduction.
Following these steps, we normalized the data between 1 and -1 in order to ensure that certain variables of different units would not have a disproportionate effect on our unsupervised and supervised learning models.

Both the PCA and T-SNE methods were applied to further reduce the dimensions of our data for both the balanced and unbalanced data, into both 2D and into 3D, so that the data could be better visualized. We extracted the explained variance of the PCA method to understand the information we retained after reducing the dimensions.
The processed data are visualized in 2D and 3D using both T-SNE methods (Figure 3 and 3.1 in results) and PCA (Figure 4 and 4.1 in results).
The explained variance of different (and cumulative) principle component indexes for PCA is plotted in Figure 5 in results.

### Unsupervised Learning

Given that our data has 10 possible combinations of preprocessing as specified in data preprocessing section above, we ran two clustering algorithms, K-Means and Gaussian Mixture Modeling on these combinations to discover the approaches that were most effective in finding clusters. We quantified which models performed must successfully by calculating the Davies Bouldin and Silhouette Coefficient internal cluster evaluation measures of the clustering results.
The possible combinations of data preprocessing are as follows:
1. Unbalanced data with label encoding, filled BMI data, dropped patient id, normalization
2. Balanced data with label encoding, filled BMI data, dropped patient id, normalization
3. Unbalanced, cleaned data with 2d TSNE
4. Balanced, cleaned data with 2d TSNE
5. Unbalanced, cleaned data with 2d PCA
6. Balanced, cleaned data with 2d PCA
7. Unbalanced, cleaned data with 3d TSNE
8. Balanced, cleaned data with 3d TSNE
9. Unbalanced, cleaned data with 3d PCA
10. Balanced, cleaned data with 3d PCA
We analyzed the preprocessed datasets using two unsupervised clustering analysis approaches for expectation maximization. First we clustered using K-Means, and then with Gaussian Mixture Modeling (GMM). 
Then we calculated the Davies Bouldin and Silhouette Coefficients for each of these clusters. 


## Results

### Data Preprocessing

After label encoding and filling in missing BMI data, the correlation heatmap between features and targets is plotted and shown in Figure 1. Due to the expected low correlation value between the “id” and the target ”stroke” we dropped this feature.

<img src="images/correlationHeatMap.png" style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 60%;"/>
<p style="text-align: center;">Figure 1</p>

After dropping the “id” feature, we performed SMOTE to balance the data. The original data contains 4861 negative cases and only 249 positive cases. The balanced data oversample at the adjacent of the minority (positive) data points to have the same number of data points as the majority (negative) data (Figure 2).

<img src="images/smote.png" style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 60%;"/>
<p style="text-align: center;">Figure 2</p>

The processed data are visualized in 3D using both T-SNE and PCA methods in Figures 3 and 4. The red X represents a positive data point, while the green dot represents a negative data point.

<img src="images/2dTSNEBal.jpg" style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 60%;"/>
<p style="text-align: center;">Figure 3 - 2D visualized data using PCA</p>

<img src="Midterm Report/TSNE1.gif" style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 60%;"/>
<p style="text-align: center;">Figure 3.1 - 3D visualized data using T-SNE</p>

<img src="images/2dPCABal.jpg" style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 60%;"/>
<p style="text-align: center;">Figure 4 - 2D visualized data using PCA</p>

<img src="Midterm Report/PCA1.gif" style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 60%;"/>
<p style="text-align: center;">Figure 4.1 - 3D visualized data using PCA</p>

The explained variance of different (and cumulative) principle component indexes is plotted in Figure 5.

<img src="images/PCAVariance.png" style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 60%;"/>
<p style="text-align: center;">Figure 5 - PCA Explained Variance</p>



## Discussion

### Data Preprocessing

From the correlation heat map (Figure 2), we can see that almost all the features are fairly independent of each other, except for “ever_married” and “age” which agree and that reflects our intuition. Even so, the correlation between these two features is only 0.68, and as such we chose to keep both features. Looking at the last row of the correlation matrix, we see both the “id” and “gender” have a relatively low correlation with our target value “stroke.” The “id” stands for a random number given to each patient and intuitively is not relative to the chance of getting a stroke. Thus, this feature is dropped with confidence. Although gender also shows a statistically low correlation to target, we still decided to keep it and leave for further steps.
In the PCA explained variance, we can see that to keep over 90% of the information, we need to maintain 9 dimensions from our 11 features. The success of clustering with 9 dimensions is very difficult to visualize. As such, we took multiple approaches to our clustering analysis to visualized both dimension reduced and full-dimension clustering results as well as with balanced and unbalanced data. By reducing to 3 dimensions with PCA, only 40% of data variance is maintained. It is clear in the data visualizations that the different targets are heavily overlapping. This can be explained by the fact that our features have very little correlation and are also all important risk factors for stroke. Thus, by reducing dimensions, important information from certain features is lost. Given this knowledge and the obvious overlapping of target values in the visualizations, unsupervised learning in 3D or lower dimension is very unlikely to distinguish the two targets. 

### Unsupervised Learning


### KMeans Clustering
We conducted the K-Means algorithm on both the original dataset and the dataset after T-SNE and PCA. The most important aspect of using this algorithm was to determine if the resulting clusters were useful for classifying specific attributes, in this case stroke risk, for specific groups of people.
The elbow method was used to determine the optimal number of clusters for the K-Means algorithm, which estimates the improvement for the addition of each cluster. Then we ran the K-Means algorithm for each of these optimal number of clusters.

### K-Means and GMM Results
Our data was preprocessed with 10 different combinations:
We performed the elbow method for GMM only on the balanced dataset of 10 features and used this number of clusters for the remaining preprocessed data:
<img src="images/GMMElbow.jpg" width="250"/>

### Unbalanced data (with label encoding, filled in missing data, and dropped patient id)
Elbow Method:
<img src="images/unbalancedDataElbow.jpg" width="250"/>
Optimal Clusters = 5

K-Means Cluster Evaluation:

### Balanced data (with label encoding, filled in missing data, and dropped patient id)
Elbow Method:
<img src="images/balancedDataElbow.jpg" width="250"/>
Optimal Clusters = 5

K-Means Cluster Evaluation:

GMM:

GMM Cluster Evaluation:

### 2D TSNE unbalanced data
Elbow Method:
<img src="images/2dTSNEUnbalancedElbow.jpg" width="250"/>
Optimal Clusters = 5

K-Means:
<img src="images/kmeans2dTSNEUnb.jpg" width="250"/>

K-Means Cluster Evaluation:

GMM:

GMM Cluster Evaluation:

### 2D TSNE balanced data
Elbow Method:
<img src="images/2dTSNEBalancedElbow.jpg" width="250"/>
Optimal Clusters = 5

K-Means:
<img src="images/kmeans2dTSNEBal.jpg" width="250"/>

K-Means Cluster Evaluation:

GMM:

GMM Cluster Evaluation:

### 3D TSNE unbalanced data
Elbow Method:
<img src="images/3dTSNEUnbalancedElbow.jpg" width="250"/>
Optimal Clusters = 7

K-Means:
<img src="images/kmeans3dTSNEUnb.jpg" width="250"/>

K-Means Cluster Evaluation:

GMM:

GMM Cluster Evaluation:

### 3D TSNE balanced data
Elbow Method:
<img src="images/3dTSNEBalancedElbow.jpg" width="250"/>
Optimal Clusters = 7

K-Means:
<img src="images/kmeans3dTSNEBal.jpg" width="250"/>

K-Means Cluster Evaluation:

GMM:

GMM Cluster Evaluation:

### 2D PCA unbalanced data
Elbow Method:
<img src="images/2dPCAUnbalancedElbow.jpg" width="250"/>
Optimal Clusters = 5

K-Means:
<img src="images/kmeans2dPCAUnb.jpg" width="250"/>

K-Means Cluster Evaluation:

GMM:

GMM Cluster Evaluation:

### 2D PCA balanced data
Elbow Method:
<img src="images/2dPCABalancedElbow.jpg" width="250"/>
Optimal Clusters = 5

K-Means:
<img src="images/kmeans2dPCABal.jpg" width="250"/>

K-Means Cluster Evaluation:

GMM:

GMM Cluster Evaluation:

### 3D PCA unbalanced data
Elbow Method:
<img src="images/3dPCAUnbalancedElbow.jpg" width="250"/>
Optimal Clusters = 5

K-Means:
<img src="images/kmeans3dPCAUnb.jpg" width="250"/>

K-Means Cluster Evaluation:

GMM:

GMM Cluster Evaluation:

### 3D PCA balanced data
Elbow Method:
<img src="images/3dPCABalancedElbow.jpg" width="250"/>
Optimal Clusters = 7

K-Means:
<img src="images/kmeans3dPCABal.jpg" width="250"/>

K-Means Cluster Evaluation:

GMM:

GMM Cluster Evaluation:


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
