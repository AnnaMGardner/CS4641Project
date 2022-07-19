import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# load data (from the same folder)
data_df = pd.read_csv('balanced.csv')

# split data into train and test
training_data_df, testing_data_df = train_test_split(data_df, test_size=0.2, random_state=25)

print(f"No. of training examples: {training_data_df.shape[0]}")
print(f"No. of testing examples: {testing_data_df.shape[0]}")

training_data = training_data_df.to_numpy()
X = training_data[:, :-1]
y = training_data[:, -1]

test_data = training_data_df.to_numpy()
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(X.shape[-1],), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
result = model.fit(X, y, epochs=50, batch_size=10)

loss, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))

plt.plot(result.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

# make class predictions with the model
predictions = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test.astype(int), predictions))

plt.show()