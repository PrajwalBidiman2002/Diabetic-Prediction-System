# %%
"""
# Diabete Disease Predicition
"""

# %%
"""
# 1 Get Data                             
# 2 preprocessing data
# 3 Train test split
# 4 Train and evualivate Model
# 5 Create Website
"""

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# %%
"""
# 1 Get Data
"""

# %%
diabete_df = pd.read_csv('diabetes.csv')
diabete_df.head(5)

# %%
"""
# 2 Proprocess Data
"""

# %%
diabete_df.shape

# %%
diabete_df['Outcome'].value_counts()

# %%
diabete_df.info()

# %%
diabete_df.describe()

# %%
diabete_df.groupby('Outcome').mean()

# %%
"""
# 3 Separate Dependent & Independ Feature into x and y
"""

# %%
X = diabete_df.drop('Outcome',axis=1)
y = diabete_df['Outcome']

# %%
print(X)

# %%
print(y)

# %%
"""
# 5 Standarization
"""

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
Standard_data = scaler.fit_transform(X)
X = Standard_data

# %%
X

# %%
"""
# 4 Train test split
"""

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=25, random_state=1)

# %%
"""
# 6 Train Model
"""

# %%
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()

# %%
model = svm.SVC(kernel='linear')

# %%
lg.fit(X_train,y_train)

# %%
# Train the SVM model as well
model.fit(X_train, y_train)

# %%
train_y_pred = lg.predict(X_train)
test_y_pred = lg.predict(X_test)

# %%
"""
# Accuracy
"""

# %%
print('Train set Accuracy :',accuracy_score(train_y_pred,y_train))
print('Test set Accuracy :',accuracy_score(test_y_pred,y_test))

# %%
# Evaluate SVM model accuracy
svm_train_pred = model.predict(X_train)
svm_test_pred = model.predict(X_test)
print('SVM Train set Accuracy:', accuracy_score(svm_train_pred, y_train))
print('SVM Test set Accuracy:', accuracy_score(svm_test_pred, y_test))

# %%
"""
# Prediction System
"""

# %%
input_data = (1,85,66,29,0,26.6,0.351,31)
input_data_nparray = np.asarray(input_data)
reshaped_input_data = input_data_nparray.reshape(1,-1)
prediction = model.predict(reshaped_input_data)

if prediction == 1:
    print('this person has a diabetes')
else:
    print("this person has not diabetes")

# %%
diabete_df.head(100)

# %%
"""
# 7 Create Website
"""

# %%
