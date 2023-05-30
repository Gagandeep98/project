#!/usr/bin/env python
# coding: utf-8

# In[1]:



# Name: Gagandeep Singh
# Student ID: 100897670

# Importing required libraries
import pandas as pd
import numpy as np

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')


# In[2]:


# Having a quick look at the data structure using head, info, and describe
print(df.head())

print(df.info())

print(df.describe())


# In[3]:


# Calculating the correlation matrix
corr_matrix = df.corr()

# Printing the correlation matrix for quality
print(corr_matrix['quality'])


# In[4]:


# Finding the value counts of the quality attribute
print(df['quality'].value_counts())


# In[5]:


from sklearn.model_selection import train_test_split

X = df.drop('quality', axis=1)
y = df['quality']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Printing the shape of the training and testing sets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[6]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Creating a SVM model with the specified hyperparameters
model1 = SVC(kernel='rbf', gamma=1, C=1)

# Training the model on the training set
model1.fit(X_train, y_train)

# Using the model to predict the quality of wine on the test set
y_predict = model1.predict(X_test)

# Printing the classification report for the test set
print(classification_report(y_test, y_predict))


# In[7]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Plotting the confusion matrix for the test results
cm = confusion_matrix(y_test, y_predict)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model1.classes_)
cm_display.plot()


# In[8]:


# Creating a second SVM model with a smaller value of C
model2 = SVC(kernel='rbf', gamma=1, C=0.01)

# Training the model on the training set
model2.fit(X_train, y_train)

# Using the model to predict the quality of wine on the test set
y_predict = model2.predict(X_test)

# Printing the classification report for the test set
print(classification_report(y_test, y_predict))


# In[9]:


# Creating a third SVM model with a larger value of C
model3 = SVC(kernel='rbf', gamma=1, C=10)

# Training the model on the training set
model3.fit(X_train, y_train)

# Using the model to predict the quality of wine on the test set
y_predict = model3.predict(X_test)

# Printing the classification report for the test set
print(classification_report(y_test, y_predict))


# In[10]:


from sklearn.model_selection import cross_val_score

# Evaluating model1 using cross validation
scores = cross_val_score(model1, X, y, cv=5)

# Printing the cross-validation scores and mean of the accuracy scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())


# In[ ]:


The model is performing well for this dataset. But the accuracy on the test set is not very good. I think that the algorithm is not performing well for the red wine dataset. There is a need to do more preprocessing to the data in order to improve the performance of the model. After loading the
winequality-white.csv dataset and comparing the results, there is not much difference in the results. There is a need to try training other machine learning algorithms such as decision trees or random forests, only then the results will improve and reach the best accuracy for this dataset.

