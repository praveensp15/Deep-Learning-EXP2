# EXPERIMENT 2 - Developing a Neural Network Classification Model

# AIM

To develop a neural network classification model for the given dataset.

# PROBLEM STATEMENT:
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.


# Neural Network Model

<img width="1045" height="801" alt="image" src="https://github.com/user-attachments/assets/d943f718-f61c-4f94-9b22-8b366204b30f" />


# DESIGN STEPS

**STEP 1:**

Import necessary libraries.

**STEP 2:**

Load the dataset "customers.csv"

**STEP 3:**

Analyse the dataset and drop the rows which has null values.

**STEP 4:**

Use encoders and change the string datatypes in the dataset.

**STEP 5:**

Calculate correlation matrix ans plot heatmap and analyse the data.

**STEP 6:**

Use various visualizations like pairplot,displot,countplot,scatterplot and visualize the data.

**STEP 7:**

Split the dataset into training and testing data using train_test_split.

**STEP 8:**

Create a neural network model with 2 hidden layers and output layer with four neurons representing multi-classification.

**STEP 9**

Compile and fit the model with the training data

**STEP 10**

Validate the model using training data.

**STEP 11**

Evaluate the model using confusion matrix.

# PROGRAM

**Name:** HANIEL REENA D R

**Register Number:** 2305001008
```
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
import pickle
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pylab as plt

# -------------------- Load & clean --------------------
customer_df = pd.read_csv('customers.csv')

customer_df_cleaned = customer_df.dropna(axis=0)

# -------------------- Encode features --------------------
categories_list = [
    ['Male', 'Female'],
    ['No', 'Yes'],
    ['No', 'Yes'],
    ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
     'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
    ['Low', 'Average', 'High']
]
enc = OrdinalEncoder(categories=categories_list)

customers_1 = customer_df_cleaned.copy()
customers_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']] = (
    enc.fit_transform(customers_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])
)

# Label-encode target (keep the encoder for inverse_transform later)
le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])

# Drop unused cols
customers_1 = customers_1.drop(['ID','Var_1'], axis=1)

# -------------------- EDA (optional) --------------------
corr = customers_1.corr(numeric_only=True)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="BuPu", annot=True)
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size', y='Spending_Score', data=customers_1)
plt.show()

# -------------------- Define X and y (FIX) --------------------
# Features = all except target; Target = Segmentation
X = customers_1.drop('Segmentation', axis=1).values
# y1 is the integer labels column (needed for OneHotEncoder); shape (n_samples, 1)
y1 = customers_1['Segmentation'].values.reshape(-1, 1)

# One-hot encode y (FIX)
try:
    one_hot_enc = OneHotEncoder(sparse_output=False)  # sklearn >=1.2
except TypeError:
    one_hot_enc = OneHotEncoder(sparse=False)         # older sklearn

y = one_hot_enc.fit_transform(y1)

# -------------------- Train/test split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=50, stratify=y
)

# -------------------- Scale (example: scale only Age column at index 2) --------------------
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:, 2].reshape(-1,1))

X_train_scaled = np.copy(X_train)
X_test_scaled  = np.copy(X_test)

X_train_scaled[:, 2] = scaler_age.transform(X_train[:, 2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,  2] = scaler_age.transform(X_test[:,  2].reshape(-1,1)).reshape(-1)

# -------------------- Model --------------------
model = Sequential([
    Dense(units=8, activation='relu', input_shape=[8]),
    Dense(units=16, activation='relu'),
    Dense(units=4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x=X_train_scaled, y=y_train,
    epochs=2000, batch_size=256,
    validation_data=(X_test_scaled, y_test),
    verbose=0
)

# -------------------- Metrics & plots --------------------
metrics = pd.DataFrame(history.history)
metrics[['loss','val_loss']].plot(title="Loss")
plt.show()

pred_proba = model.predict(X_test_scaled)
x_test_predictions = np.argmax(pred_proba, axis=1)
y_test_truevalue = np.argmax(y_test, axis=1)

print(confusion_matrix(y_test_truevalue, x_test_predictions))
print(classification_report(y_test_truevalue, x_test_predictions))

# -------------------- Save model & artifacts --------------------
model.save('customer_classification_model.h5')
with open('customer_data.pickle', 'wb') as fh:
    pickle.dump([X_train_scaled, y_train, X_test_scaled, y_test,
                 customers_1, customer_df_cleaned, scaler_age, enc, one_hot_enc, le], fh)

# -------------------- Load & single prediction demo --------------------
model = load_model('customer_classification_model.h5')
with open('customer_data.pickle', 'rb') as fh:
    [X_train_scaled, y_train, X_test_scaled, y_test,
     customers_1, customer_df_cleaned, scaler_age, enc, one_hot_enc, le] = pickle.load(fh)

x_single_prediction = np.argmax(model.predict(X_test_scaled[1:2, :]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```
**Dataset Information**

<img width="1265" height="364" alt="image" src="https://github.com/user-attachments/assets/4d8547b0-2203-43c5-98eb-0ba496da3ffa" />




# OUTPUT

**Training Loss, Validation Loss Vs Iteration Plot:**

<img width="547" height="435" alt="image" src="https://github.com/user-attachments/assets/1cb07a18-8846-4c3c-a13d-f199267b844a" />


**Classification Report**

<img width="523" height="206" alt="image" src="https://github.com/user-attachments/assets/a0428e5f-f3d1-409d-b6a3-412486221dee" />




**Confusion Matrix**

<img width="259" height="94" alt="image" src="https://github.com/user-attachments/assets/c50c81d0-4e29-4859-9120-e5c81045234f" />


**New Sample Data Prediction**

<img width="822" height="300" alt="image" src="https://github.com/user-attachments/assets/c59cf4a9-830c-4c22-8ffa-a920412af385" />


**RESULT**

Thus a neural network classification model is developed for the given dataset.

