import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, hidden_dim=1024,num_filters=384,attention_hop=20,num_classes=8,attention=True):
        super(Net, self).__init__()
        self.hidden_dim=hidden_dim
        self.attention_hop = attention_hop
        self.attention = attention
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=(40,8))#, padding=3)

        self.conv2 = nn.Conv1d(1, num_filters, kernel_size=(40,16))#, padding=7)

        self.conv3 = nn.Conv1d(1, num_filters, kernel_size=(40,32))#, padding=15)

        self.conv4 = nn.Conv1d(1, num_filters, kernel_size=(40,64))#, padding=31)

        self.max_pool = nn.MaxPool1d(kernel_size=7, stride=7)
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(num_filters, self.hidden_dim)
        if self.attention:
            self.fc2 = nn.Linear(self.hidden_dim, self.attention_hop)
            self.fc3 = nn.Linear(self.attention_hop, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        c1 = torch.flatten(F.relu(self.conv1(x)),start_dim=2)
        c2 = torch.flatten(F.relu(self.conv2(x)),start_dim=2)
        c3 = torch.flatten(F.relu(self.conv3(x)),start_dim=2)
        c4 = torch.flatten(F.relu(self.conv4(x)),start_dim=2)
        # print("x",x.shape)
        # print("c1",c1.shape)
        # print("c2",c2.shape)
        # print("c3",c3.shape)
        # print("c4",c4.shape)
        #print("Applying max pool")
        c1 = self.max_pool(c1)
        c2 = self.max_pool(c2)
        c3 = self.max_pool(c3)
        #c4 = self.max_pool(c4).squeeze_(2)
        if c4.shape[-1]>7:
            c4 = self.max_pool(c4)
        # print("c1",c1.shape)
        # print("c2",c2.shape)
        # print("c3",c3.shape)
        # print("c4",c4.shape)

        x = torch.cat((c1,c2,c3,c4),dim=2).transpose(2,1)
        #print("flatten",x.shape)
        x = self.dropout(x)
        if self.attention:
            xw1 = torch.tanh(self.fc1(x))
            A = F.softmax(self.fc2(xw1),dim=-1)
            #print(A.shape,x.shape)

            E = torch.matmul(x.transpose(2,1),A)
            #print("E",E.shape)
            x = self.fc4(self.fc3(E))
            #print("x",x.shape)
        else:
            x=F.leaky_relu(self.fc4(F.leaky_relu(self.fc1(x))))
        return x
