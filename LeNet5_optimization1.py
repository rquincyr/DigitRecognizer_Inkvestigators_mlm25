#!/usr/bin/env python
# coding: utf-8

# In[3]:


# LeNet-5 optimization try 1


# In[5]:


# install torch
get_ipython().system('pip install torch torchvision torchaudio')


# In[8]:


import torch
print(torch.__version__)


# In[22]:


# importing
import torch.nn as nn
import torch.nn.functional as F  # For activation functions and pooling
import torch.optim as optim  # # For optimizers (SGD, Adam, etc.)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
import pandas as pd
import numpy as np


# In[24]:


# Use GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")


# In[27]:


# load the data
# data is in this folder
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")


# In[30]:


pwd


# In[34]:


train_df.head()


# In[36]:


# separating out the labels from the pixel data features
X_train = train_df.drop('label', axis=1).values   # shape: (42000, 784)
y_train = train_df['label'].values                # shape: (42000,)
X_test = test_df.values                           # shape: (28000, 784)


# In[38]:


# need to convert to PyTorch tensors and reshape
# converts to float32 and normalizes to [0,1]
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0


# In[40]:


# check the mean and standard deviation from the normalization
# should be mean 0 and std of 1
mean = X_train_tensor.mean().item()
std = X_train_tensor.std().item()
print(f"Training data mean: {mean:.4f}, std: {std:.4f}")


# In[48]:


# split training data into 80 train and 20 validate from train
train_size = int(0.8*len(X_train_tensor))
val_size = len(X_train_tensor)-train_size
train_dataset, val_dataset = random_split(
    TensorDataset(X_train_tensor, y_train_tensor),
    [train_size, val_size]
)


# In[50]:


# writing up the data loaders 
# allows for batches of specified size to make training more efficient
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader  = DataLoader(X_test_tensor, batch_size=128, shuffle=False)


# In[52]:


# set up the optimized LeNet model
# differences from the baseline model include using: 
# ReLU instead of tanh, MaxPool instead of AvgPool, BatchNorm for stability, and Dropout for normalization
class LeNet5_Optimized(nn.Module):
    def __init__(self):
        super(LeNet5_Optimized, self).__init__()
        # Convolutional layer 1: 1 input channel, 6 output feature maps
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(6)            # BatchNorm stabilizes learning
        
        # Convolutional layer 2: 6 input, 16 output channels
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2   = nn.BatchNorm2d(16)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)     # Flattened conv output → 120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)              # 10 digits → 10 output logits
        
        # Dropout regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # ---- Conv Block 1 ----
        # Conv → BatchNorm → ReLU → MaxPool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # ---- Conv Block 2 ----
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # ---- Flatten ----
        x = x.view(-1, 16 * 5 * 5)
        
        # ---- Fully Connected Layers ----
        x = F.relu(self.fc1(x))
        x = self.dropout(x)             # Drop neurons to avoid overfitting
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                 # Output raw logits
        return x


# In[54]:


# the training set up
model = LeNet5_Optimized().to(device)

# CrossEntropyLoss combines softmax + negative log likelihood.
criterion = nn.CrossEntropyLoss()

# Adam optimizer adapts learning rates for each parameter.
# weight_decay adds L2 regularization for further generalization.
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler gradually decreases the LR every 5 epochs by 20%.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)


# In[58]:


# training the model
epochs=50
patience=5
best_val_acc=0.0
epochs_no_improve=0


for epoch in range(epochs):
    model.train()                        # Enable dropout + batchnorm updates
    running_loss = 0.0
    
    # ---- Training phase ----
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()            # Reset gradient buffers
        output = model(data)             # Forward pass
        loss = criterion(output, target) # Compute cross-entropy loss
        loss.backward()                  # Backpropagate gradients
        optimizer.step()                 # Update weights
        
        running_loss += loss.item()      # Accumulate batch loss
    
    scheduler.step()                     # Adjust learning rate schedule if using scheduler
    
    # ---- Validation phase ----
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_acc = 100 * correct / total
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_acc:.2f}%")
    
    # ---- Early Stopping Logic ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        # Save the best model weights
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
    
    # Stop if validation accuracy hasn't improved for 'patience' epochs
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Load the best model weights before generating predictions
model.load_state_dict(best_model_state)
print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")


# In[60]:


# generate csv for submission
model.eval()
predictions = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)  # Convert logits → predicted labels
        predictions.extend(predicted.cpu().numpy())

# Create submission DataFrame (format required by Kaggle)
submission = pd.DataFrame({
    "ImageId": np.arange(1, len(predictions) + 1),
    "Label": predictions
})

# Save predictions to CSV
submission.to_csv("LeNet5_Optimized_submission_LW.csv", index=False)
print("Submission file 'LeNet5_Optimized_submission_LW.csv' created successfully.")


# In[5]:


# convert to .py file
get_ipython().system('jupyter nbconvert --to script LeNet-5_optimized_model_LW.ipynb')


# In[ ]:




