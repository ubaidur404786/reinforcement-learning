
import torch.nn as nn
import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset
import glob

from torch.optim import Adam
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import random
import matplotlib.pyplot as plt

EPOCH =10
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = CrossEntropyLoss()
criterion = BCELoss()

class Dataset(Dataset):

    def __init__(self, path, train=True):

        # Fetching saved files
        self.files = glob.glob(path)
        if train:
            self.files = self.files[0:int(len(self.files)*0.75)]
        else:
            self.files = self.files[int(len(self.files)*0.75):]

        # Counting the number of sample for each label
        label_counts = np.zeros((3))
        for idx in range(len(self.files)):
            label = int(self.files[idx][-5])
            if label>0 : label -= 1
            label_counts[label]+=1
        min_class = np.min(label_counts)

        # Balancing labels
        new_files = []
        label_counts = torch.tensor([0, 0, 0])
        for f in range(len(self.files)):
            label = int(self.files[f][-5])
            if label>0 : label -= 1
            if label_counts[label]<min_class:
                new_files.append(self.files[f])
                label_counts[label]+=1
        self.files = new_files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        frames = np.load(self.files[idx])
        label = int(self.files[idx][-5])

        if label==0:
            label = np.array([1, 0, 0])
        elif label==2:
            label = np.array([0, 1, 0])
        elif label==3:
            label = np.array([0, 0, 1])

        frames = frames[0]/2+frames[1]
  
        frames[frames==216] = 0.
        frames[frames!=0] = 1.

        frames = torch.tensor([frames])
        return frames, label
    

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),

    torch.nn.Flatten(),
    torch.nn.Linear(288, 3)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataset = Dataset("./pong_game/data/*")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
test_dataset = Dataset("./pong_game/data/*", train=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

label_counts = torch.tensor([0, 0, 0])
for data, label in tqdm(train_dataloader):
    for b in range(len(label)):
        label_counts[torch.argmax(label[b])]+=1


for e in range(EPOCH):
    model.train()

    mean_train_loss = 0
    cpt = 0
    for data, label in tqdm(train_dataloader):
        pred = model(data.to(device).float())
        pred = torch.nn.functional.softmax(pred)
      
        loss = criterion(pred, label.to(device).float())
        loss.backward()
        optimizer.step()
        mean_train_loss+=loss.sum()
        cpt+=1
    print("Train loss:", mean_train_loss/(cpt*batch_size))

    
    model.eval()
    mean_test_loss = 0
    cpt = 0
    with torch.no_grad():
        for data, label in tqdm(test_dataloader):
            pred = model(data.to(device).float())
            pred = torch.nn.functional.softmax(pred)
        
            loss = criterion(pred, label.to(device).float())
            mean_test_loss+=loss.sum()
            cpt+=1
        print("Test loss:", mean_test_loss/(cpt*batch_size))

torch.save(model, "./pong_game/model.pt")
       
     
    




