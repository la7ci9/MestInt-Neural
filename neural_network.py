# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Part 1 - Data Preprocessing

def preprocess_data(data):
    #Replace NaN values with the mean of the column
    dataset.fillna(dataset.mean(), inplace=True)

    #Create X features and y output values
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values

    #Encode 'male/female' to 0/1
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X[:, 1] = le.fit_transform(X[:, 1])

    # One Hot Encoding the "Pclass" column
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X, y

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
print(dataset.head())
X_train, y_train = preprocess_data(dataset)

test_dataset = pd.read_csv('test.csv')
test_dataset = test_dataset[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
X_test, y_test = preprocess_data(test_dataset)

# TODO Create a NN with 3 layers of the size (18,12,6), 'relu' activation for the hidden layers, 'sigmoid' for the output layer

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))

ann.add(tf.keras.layers.Dense(units=12, activation='relu'))

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation


#print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)


# Predicting the Test set results
ann.evaluate(X_test, y_test)
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)