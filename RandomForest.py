import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# load data (from the same folder)
data_df = pd.read_csv('processed.csv')


# split data into train and test
training_data_df, testing_data_df = train_test_split(data_df, test_size=0.2, random_state=25)

print(f"No. of training examples: {training_data_df.shape[0]}")
print(f"No. of testing examples: {testing_data_df.shape[0]}")

training_data = training_data_df.to_numpy()
X = training_data[:, :-1]
y = training_data[:, -1]

test_data = testing_data_df.to_numpy()
X_test = test_data[:, :-1]
y_test = test_data[:, -1]


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=500 , max_depth = 500)

clf.fit(X,y)

predictions = clf.predict(X_test)

print(len(predictions))
print(classification_report(y_test.astype(int), predictions))

plot_confusion_matrix(clf, X_test, y_test)
plt.show()