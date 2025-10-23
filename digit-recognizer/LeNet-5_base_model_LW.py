#!/usr/bin/env python
# coding: utf-8

# In[1]:


# LeNet-5 base model work


# In[6]:


# working with the parameters from the original 1998 paper
# will then work on iterative optimization of different areas


# In[12]:


# install torch
get_ipython().system('pip install torch torchvision torchaudio')


# In[3]:


import torch
print(torch.__version__)


# In[1]:


# importing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# In[6]:


# Step 1: Load MNIST with only ToTensor() (no normalization)
raw_transform = transforms.Compose([
    transforms.ToTensor()
])


# In[9]:


pwd


# In[13]:


import pandas as pd


# In[15]:


# need to read in the kaggle csv file, ensuring that the notebook doesn't use the pytorch dataset (which is different)

# Load Kaggle train.csv
train_df = pd.read_csv('train.csv')  # update path if needed

# Separate features and labels 
X_train = train_df.drop('label', axis=1).values   # shape: (42000, 784)
y_train = train_df['label'].values                # shape: (42000,)

# Convert to PyTorch tensors and reshape 
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
y_train_tensor = torch.tensor(y_train, dtype=torch.long)


# In[19]:


# Wrap in TensorDataset and DataLoader
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# In[27]:


# going to normalize the data, need the mean and std for z score normaliztion
# only on the test set so that there is no test set data leakage
mean = X_train_tensor.mean().item()
std = X_train_tensor.std().item()
print(f"Mean: {mean:.4f}, Std: {std:.4f}")


# In[ ]:





# In[ ]:





# In[54]:


## Standard normalization


# In[56]:


# ignore all of the commented out commands below, these are using tensor transform
# this only applies to one sample, [0], and i want to apply it to everything
# i am going to do this manually now below the commented out information


# In[58]:


# z score normalization with calculated mean and std of the normalized (0 to 1) scores
# defines what the transformation is
#transform = transforms.Normalize((mean,), (std,))


# In[ ]:





# In[ ]:





# In[45]:


# Apply to a single batch for testing:
#sample = X_train_tensor[0]  # shape [1,28,28]
#normalized_sample = transform(sample)


# In[47]:


# check the mean and std
# should be 0 and 1
#print(f"Sample mean after normalization: {normalized_sample.mean():.4f}")
#print(f"Sample std after normalization:  {normalized_sample.std():.4f}")


# In[60]:


# Apply z-score normalization to the entire training set
X_train_normalized = (X_train_tensor - mean) / std


# In[62]:


# Wrap the normalized data in a TensorDataset and DataLoader
train_dataset = TensorDataset(X_train_normalized, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# In[64]:


# Check that normalization worked across the whole dataset
print(f"Mean after normalization: {X_train_normalized.mean():.4f}")
print(f"Std after normalization:  {X_train_normalized.std():.4f}")


# In[66]:


# this also created the train_loader data loader to use to feed into the CNN


# In[68]:


# Define LeNet-5 Model
# -------------------------------
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # flatten 16*4*4 feature maps into 120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)           # 10 output classes for digits 0-9

    def forward(self, x):
        # Forward pass through the network
        x = torch.tanh(self.conv1(x))          # apply first conv layer + tanh activation
        x = F.avg_pool2d(x, 2)                 # average pooling with 2x2 kernel
        x = torch.tanh(self.conv2(x))          # second conv layer + tanh
        x = F.avg_pool2d(x, 2)                 # second average pooling
        x = x.view(-1, 16 * 4 * 4)             # flatten tensor for fully connected layers
        x = torch.tanh(self.fc1(x))            # first FC layer + tanh
        x = torch.tanh(self.fc2(x))            # second FC layer + tanh
        x = self.fc3(x)                        # output layer (logits)
        return F.log_softmax(x, dim=1)         # log softmax for classification


# In[70]:


# Setup for Training
# -------------------------------
# Choose device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move it to the chosen device
model = LeNet5().to(device)

# Define the optimizer: SGD with learning rate 0.01 and momentum 0.9
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Define the loss function: CrossEntropyLoss is standard for multi-class classification
criterion = nn.CrossEntropyLoss()


# In[72]:


# Training Loop
# -------------------------------
epochs = 10  # number of training epochs

for epoch in range(epochs):
    model.train()  # set model to training mode
    for data, target in train_loader:  # iterate over batches
        data, target = data.to(device), target.to(device)  # move batch to GPU/CPU
        
        optimizer.zero_grad()       # reset gradients from previous step
        output = model(data)       # forward pass
        loss = criterion(output, target)  # compute loss
        loss.backward()            # backpropagation
        optimizer.step()           # update model parameters
    
    print(f"Epoch {epoch+1}/{epochs} done.")  # progress message


# In[ ]:





# In[78]:


# Normalizing the test dataset
test_df = pd.read_csv('test.csv')   
X_test = test_df.values
# For Kaggle, test.csv has no labels. 
# If you have validation labels, replace this line with y_test from validation CSV
y_test = torch.zeros(X_test.shape[0], dtype=torch.long)  # placeholder if no labels
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
y_test_tensor = y_test  # if labels available

# Apply Z-score normalization to all images
X_test_normalized = (X_test_tensor - mean) / std  # use training mean/std

# load into test dataset loader
# Wrap the normalized test data and labels into a TensorDataset
test_dataset = TensorDataset(X_test_normalized, y_test_tensor)
# DataLoader for the test set; no shuffling needed
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# In[82]:


# Evaluation
model.eval()  # evaluation mode
test_predictions = []

with torch.no_grad():  # no gradients needed
    for data, _ in test_loader:  # labels aren’t in test.csv
        data = data.to(device)
        output = model(data)
        pred = output.argmax(dim=1)  # get predicted class
        test_predictions.extend(pred.cpu().numpy())


# In[86]:


# generating a submission file to submit to kaggle for accuracy predictions
# i did not make an 80/20 split on the training data in the set, so i am just applying it to the test data
submission = pd.DataFrame({
    "ImageId": range(1, len(test_predictions)+1),
    "Label": test_predictions
})

submission.to_csv("LeNet-5_baseline_fulltrain_submission_LW_101925.csv", index=False)
print("LeNet-5_baseline_fulltrain_submission_LW_101925.csv ready for Kaggle!")


# In[ ]:




