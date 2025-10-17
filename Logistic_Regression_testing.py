#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Logistic Regression


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


# Machine learning modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


# In[7]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
train.head()


# In[9]:


# Split features and target
X = train.drop('label', axis=1)
y = train['label']

# Normalize pixel values
X = X / 255.0


# In[11]:


# Split training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=4
)


# In[13]:


print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")


# In[15]:


# Initialize model
log_reg = LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=1000,
    n_jobs=-1  # use all available CPU cores
)

print("Training logistic regression model...")
log_reg.fit(X_train, y_train)
print("✅ Model training complete.")


# In[17]:


# Predict on validation set
y_pred = log_reg.predict(X_val)

# Metrics
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}\n")
print("Classification Report:\n", classification_report(y_val, y_pred))


# In[19]:


cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d', colorbar=False)
plt.title("Confusion Matrix")
plt.show()


# In[21]:


# Get misclassified indices
misclassified_idx = np.where(y_val != y_pred)[0]

# Show a few
plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified_idx[:9]):
    image = np.array(X_val.iloc[idx]).reshape(28, 28)
    plt.subplot(3, 3, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"True: {y_val.iloc[idx]}, Pred: {y_pred[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[23]:


# Normalize test data
test = test / 255.0

# Predict
predictions = log_reg.predict(test)

# Create submission CSV
submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})

submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")


# In[25]:


# made and submitted this file of the test data
# the result was Score: 0.91825 on Kaggle


# In[27]:


get_ipython().system('jupyter nbconvert --to script Logistic_Regression_test.ipynb')


# In[ ]:




