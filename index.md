## Stroke Prediction

![stroke pic](Stroke_Web.png)
# Original dataset

| Total Number of Patient | Total Number of Features  | Stroke or Not? |
| ----------------------- | ------------------------- | -------------- |
|           5110          |              9            |       Y/N      |

| Patient ID | Gender | Age | Hypertension | Heart Disease | Ever Married | Work Type | Residence Type | Average Glucose Level | BMI | Smoking Status |
| ---------- | ------ | --- | ------------ | ------------- | ------------ | --------- | -------------- | --------------------- | --- | -------------- |
|  67-72940  |   F/M  | 0-82|      Y/N     |      Y/N      |     Y/N      |     4     |  Urban/Rural   |         55-271        |10-97|        4       |


You can use the [editor on GitHub](https://github.com/AnnaMGardner/CS4641Project/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

600 words max
Good example: https://jiajunmao.github.io/4641-Project/
Markdown: (https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

## Summary (infographic with project goal) (Yawen)
[Stroke.pdf](https://github.com/AnnaMGardner/CS4641Project/files/8846563/Stroke.pdf)

## Introduction (Yawen)

*Yawen 2022-06-05 6:53pm edits:*  

As one of the popular diseases, stroke affects the arteries within and leading to the brain. Globally, stroke is the second leading cause of death, accounting for approximately 11% of all deaths, according to the World Health Organization (WHO). There are many factors that can be used to predict a patient's risk of stroke, including high blood pressure, smoking, diabetes, high cholesterol levels, heavy drinking, high salt and fat diets, and lack of exercise. And most importantly, older people are more easily to have stroke than younger people. In addition, those who have already suffered a stroke are at greater risk of experiencing another. Therefore, it becomes possible for us to predict whether a person has high possibility to get stroke or not based on previous dataset. Our results can also remind people who have similar conditions to change their life styles to avoid stroke.

*end of edit*  

## Methods (Anna, Zhenming)

*Zhenming 2022-06-06 5:52pm edits:*  

The stroke prediction dataset [1] will be used in this project. There are a total of 5110 row (number of samples) and 12 columns with 11 features and one target column. The feature columns include physiological information believed to be relative to the chance of getting a stroke. The feature column contains both string and an integer value. We will use label coding to convert any string value to an integer value for better interpretation of the dataset. The target column is a 1-D array of boolean values indicating whether stroke risk is identified.  

The raw data is unbalanced: 249 data points identify the chance of stroke, and 4821 data points have no stroke risk. We will preprocess the dataset with the synthetic minority oversampling technique (SMOTE) to balance the data [6]. The processed data will be split into two segments, with 80%  for training and the remaining for testing.

To analyze the dataset, we will start with t-distributed stochastic neighbor embedding, an unsupervised learning method to visualize high dimension data to find the potential correlation between different features. Followed by supervised learning, aiming to diagnose and predict stroke risk.

Common supervised algorithms used for stroke prediction include Decision Tree, Voting Classifier[2], and Random Forecast, Logistic Regression [3]. Throughout this project, we will construct alternative models based on the above algorithms and compare the accuracy and precision of each method. 

*end of edit*  


## Results (predicion of outcome) (Emma)
(what results are you trying to achieve? )
*Anna 2022-07-06 2:27pm edits:*
The expected outcome of our dataset stochastic neighbor embedding is that there will exist clusters of similar patient datapoints. These clusters represent patients who have similar health feature values. Given these similarities, one or more of these clusters may represent patients with high risk of stroke. The expected relationship between health features and stroke likelihood is expected to reflect known stroke risk factors including age, diabetes, and hypertension [5].
Additionally, the expected outcome of a supervised algorithm for stroke predicion is a binary classification and prediction of the data point's stroke value. Given the relatively small size of the dataset and the need for synthetic minority data creation, we aim for an accuracy of 80% for our test data. 

## Discussion (Emma)
(best outcome, what it would mean, what is next.....
Reaching a prediction accuracy of over 80% for stroke risk would mean that from a simple set of health measurements this model can identify a person at high risk for stroke. This model would then be able to identify which individuals should take preventative measures for strokes. 
Additionally, with a trained supervised model of high accuracy stroke prediction, we aim to create a stroke risk calculator which can non-medically predict the likelikood of patient stroke. 

*end of edit*
## References (at least 3 - peer reviewed) (Anna, Zhenming)

*Zhenming 2022-06-04 10:21pm edit:*  

***dataset:***  
[1]“Stroke prediction dataset,” [Online]. Available: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset.  

***ML 1: This paper used the same dateset we are using. LOL. This one use be very useful!***  
[2] Tahia Tazin, Md Nur Alam, Nahian Nakiba Dola, Mohammad Sajibul Bari, Sami Bourouis, Mohammad Monirujjaman Khan, "Stroke Disease Detection and Prediction Using Robust Learning Approaches", Journal of Healthcare Engineering, vol. 2021, Article ID 7633381, 12 pages, 2021. https://doi.org/10.1155/2021/7633381.  

***ML2:***  
[3] JoonNyung Heo, Jihoon Yoon, Hyungjong Park, Young Kim, Hyo Suk Nam, Ji Hoe Heo. "Machine Learning–Based Model for Prediction of Outcomes in Acute Stroke". Stroke. 50. 1263–1265, 2019 http://doi.org/10.1161/STROKEAHA.118.024293.  

***Medical: Table 2 gives most common symptoms and signs of stroke.***  
[4] Yew, Kenneth S, and Eric Cheng. “Acute stroke diagnosis.” American family physician vol. 80,1 (2009): 33-40. http://www.ncbi.nlm.nih.gov/pmc/articles/pmc2722757/  

[5] Boehme, Amelia K et al. “Stroke Risk Factors, Genetics, and Prevention.” Circulation research vol. 120,3 (2017): 472-495. doi:10.1161/CIRCRESAHA.116.308398

*end of edit*

*Anna 2022-07-06 2:27pm edit:*

***Preprocessing:***
[6] Chawla, Nitesh & Bowyer, Kevin & Hall, Lawrence & Kegelmeyer, W.. (2002). SMOTE: Synthetic Minority Over-sampling Technique. J. Artif. Intell. Res. (JAIR). 16. 321-357. 10.1613/jair.953. 

*end of edit*


Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
