## Stroke Prediction

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

*Zhenming 2022-06-04 10:21pm edits:*  

The stroke prediction dataset [1] will be used in this project. There are a total of 5110 row (number of samples) and 12 columns with 11 features and one target column. The feature columns include physiological information believed to be relative to the chance of getting stroke and the target column indicates whether a stroke risk is identified. In this project, we will first use unsupervised learning to seek potential pattern between the 11 features, followed by supervised learning aiming to detect and predict the risk of having stroke.   

Some common machine learning algorithms used for stroke prediction include Decision Tree, Voting Classifier[2], and Random Forecast, Logistic Regression [3]. Throughout this project, we will construct alternative models based on the above algorithms, and compare the accuracy and precision of each method. As we further study the given data set, a data preprocess may be performed, if necessary, to remove unwanted outliers and noises for impoving the final results.

*end of edit*  


## Results (predicion of outcome) (Emma)
(what results are you trying to achieve? )
## Discussion (Emma)
(best outcome, what it would mean, what is next.....
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

*end of edit*


1 stroke prediction method from literature (method)  
1 medical predicion method (method)  
1 medical paper on strokes and risks for strokes (expected results resource) dataset itself  




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

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/AnnaMGardner/CS4641Project/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
