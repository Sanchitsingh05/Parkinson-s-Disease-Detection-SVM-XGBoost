#!/usr/bin/env python
# coding: utf-8

#  ## Installing Required Libraries

# In[8]:


# Libraries for data manipulation
import numpy as np
import pandas as pd

# Libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Libraries for preprocessing and model building
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# For XGBoost
from xgboost import XGBClassifier


# ## Data Collection and Analysis

# In[9]:


# Load the dataset
parkinsons_data = pd.read_csv('parkinsons.csv')


# In[39]:


# Display first 5 rows
print("First 5 rows of the dataset:")
parkinsons_data.head()


# In[11]:


# Dataset shape
print("\nDataset Shape (rows, columns):", parkinsons_data.shape)


# In[12]:


# Data info
print("\nDataset Info:")
print(parkinsons_data.info())


# In[13]:


# Checking missing values
print("\nMissing values in each column:")
print(parkinsons_data.isnull().sum())


# In[45]:


# Distribution of target variable
sns.countplot(x='status', data=parkinsons_data)
plt.title("Distribution of Parkinson's Disease Status (1 = Diseased, 0 = Healthy)")
plt.xlabel("Status")
plt.ylabel("Count")
plt.show()


# ## Data Pre-Processing
# 
# ### Seperating the features & Target

# In[40]:


# Splitting the data into Features and Target
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']


# In[41]:


print(X)


# In[42]:


print(Y)


# In[43]:


# Splitting into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2, stratify=Y)


# In[44]:


print(X.shape, X_train.shape, X_test.shape)


# In[18]:


# Feature Scaling
scaler = StandardScaler()
scaler.fit(X_train)


# ## Data Standardization
# 
# ### Convert all data in columns in same range, without altering the meaning that value conveys

# In[19]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # SVC
# 
# SVC stands for Support Vector Classifier, a part of the SVM family. It tries to find the optimal hyperplane that best separates the classes (e.g., Parkinson's vs. non-Parkinson's) in the feature space.

# In[20]:


# Create SVM model
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, Y_train)


# In[21]:


# Predicting on training and test data
svm_train_pred = svm_model.predict(X_train)
svm_test_pred = svm_model.predict(X_test)


# In[22]:


# Accuracy scores
print("SVM Training Accuracy:", accuracy_score(Y_train, svm_train_pred))
print("SVM Testing Accuracy:", accuracy_score(Y_test, svm_test_pred))


# In[23]:


# Classification report and confusion matrix
print("\nSVM Classification Report:\n", classification_report(Y_test, svm_test_pred))


# In[24]:


sns.heatmap(confusion_matrix(Y_test, svm_test_pred), annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#  # XGBoost
#  
# XGBoost is a powerful and efficient gradient boosting algorithm used for supervised learning. It builds an ensemble of decision trees sequentially, optimizing to reduce errors and often gives top-tier results in real-world problems.

# In[25]:


# Create XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, Y_train)


# In[26]:


# Predicting on training and test data
xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)


# In[27]:


# Accuracy scores
print("XGBoost Training Accuracy:", accuracy_score(Y_train, xgb_train_pred))
print("XGBoost Testing Accuracy:", accuracy_score(Y_test, xgb_test_pred))


# In[28]:


# Classification report and confusion matrix
print("\nXGBoost Classification Report:\n", classification_report(Y_test, xgb_test_pred))


# In[46]:


sns.heatmap(confusion_matrix(Y_test, xgb_test_pred), annot=True, fmt='d', cmap='Greens')
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[34]:


# Sample Input Data (same as before)
input_data = (223.36500,238.98700,98.66400,0.00264,0.00001,0.00154,0.00151,0.00461,0.01906,
              0.16500,0.01013,0.01296,0.01340,0.03039,0.00301,26.13800,0.447979,0.686264,
              -7.293801,0.086372,2.321560,0.098555)

# Get column names used for training
feature_names = X.columns  # X is the DataFrame you trained on

# Wrap input data in a DataFrame
input_data_df = pd.DataFrame([input_data], columns=feature_names)

# Standardize using scaler
input_data_std = scaler.transform(input_data_df)


# ## SVM Prediction

# In[37]:


# --------- SVM Prediction ----------
svm_input_pred = svm_model.predict(input_data_std)
print("\n[SVM Prediction]")
print(svm_input_pred)
if svm_input_pred[0] == 0:
    print("SVM: The person does NOT have Parkinson's Disease.")
else:
    print("SVM: The person HAS Parkinson's Disease.")


# ## XGBoost Prediction

# In[38]:


# --------- XGBoost Prediction ----------
xgb_input_pred = xgb_model.predict(input_data_std)
print("\n[XGBoost Prediction]")
print(xgb_input_pred)
if xgb_input_pred[0] == 0:
    print("XGBoost: The person does NOT have Parkinson's Disease.")
else:
    print("XGBoost: The person HAS Parkinson's Disease.")


# In[ ]:




