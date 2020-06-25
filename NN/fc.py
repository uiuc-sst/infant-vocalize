import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, input_size=201, hidden_size=5, num_classes=4, fine_tune=True, stack=False):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fine_tune=fine_tune
        self.hidden1=torch.tensor([],dtype=torch.torch.float32,device=torch.device("cuda"))
        self.hidden2=torch.tensor([],dtype=torch.torch.float32,device=torch.device("cuda"))
        self.stack=stack

    def forward(self, x):
        x=F.leaky_relu(self.fc1(x))
        out=F.leaky_relu(self.fc2(x))
        if self.stack:
            self.hidden1=torch.cat((self.hidden1,x))
            self.hidden2=torch.cat((self.hidden2,out))
        return out

    def get_hidden_nodes(self):
        return self.hidden1,self.hidden2

class threeLayerFC(nn.Module):
    def __init__(self, input_size=201, hidden_size=5, num_classes=4):
        super(threeLayerFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2), num_classes)

    def forward(self, x):
        out = self.fc3(self.fc2(self.fc1(x)))
        return out
