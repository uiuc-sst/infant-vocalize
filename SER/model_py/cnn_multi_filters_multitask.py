import torch
import torch.nn as nn
import torch.nn.functional as F

class selfAtt(nn.Module):
    def __init__(self, hidden_dim=1024,num_filters=384,attention_hop=20,num_classes=4):
        super(selfAtt, self).__init__()
        self.hidden_dim=hidden_dim
        self.attention_hop = attention_hop
        self.num_filters=num_filters
        self.num_classes=num_classes

        self.fc1 = nn.Linear(self.num_filters, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.attention_hop)

        self.fc3 = nn.Linear(self.attention_hop, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self,x):
        xw = torch.tanh(self.fc1(x))
        A = F.softmax(self.fc2(xw),dim=-1)
        E = torch.matmul(x.transpose(2,1),A)
        x = self.fc4(self.fc3(E))
        return x

class Net(nn.Module):
    def __init__(self, att_module, hidden_dim=1024,num_filters=384,attention_hop=20,num_classes=4):
        super(Net, self).__init__()
        self.hidden_dim=hidden_dim
        self.attention_hop = attention_hop
        self.num_filters=num_filters
        self.num_classes=num_classes
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=(40,8))#, padding=3)

        self.conv2 = nn.Conv1d(1, num_filters, kernel_size=(40,16))#, padding=7)

        self.conv3 = nn.Conv1d(1, num_filters, kernel_size=(40,32))#, padding=15)

        self.conv4 = nn.Conv1d(1, num_filters, kernel_size=(40,64))#, padding=31)

        self.max_pool = nn.MaxPool2d(kernel_size=(1,7), stride=7)
        self.dropout = nn.Dropout(0.2)

        self.attm0=att_module(hidden_dim,num_filters,attention_hop,num_classes)
        #self.attm1=att_module(hidden_dim,num_filters,attention_hop,num_classes)
        self.attm2=att_module(hidden_dim,num_filters,attention_hop,num_classes)
        self.attm3=att_module(hidden_dim,num_filters,attention_hop,num_classes)
        self.attm4=att_module(hidden_dim,num_filters,attention_hop,num_classes)

        self.attc0=att_module(hidden_dim,num_filters,attention_hop,num_classes)
        self.attc1=att_module(hidden_dim,num_filters,attention_hop,num_classes)
        self.attc2=att_module(hidden_dim,num_filters,attention_hop,num_classes)
        self.attc3=att_module(hidden_dim,num_filters,attention_hop,num_classes)
        self.attc4=att_module(hidden_dim,num_filters,attention_hop,num_classes)

    def forward(self, x):
        c1 = (F.relu(self.conv1(x)))
        c2 = (F.relu(self.conv2(x)))
        c3 = (F.relu(self.conv3(x)))
        c4 = (F.relu(self.conv4(x)))

        c1 = self.max_pool(c1).squeeze_(2)
        c2 = self.max_pool(c2).squeeze_(2)
        c3 = self.max_pool(c3).squeeze_(2)
        c4 = self.max_pool(c4).squeeze_(2)


        x = torch.cat((c1,c2,c3,c4),dim=2).transpose(2,1)
        x = self.dropout(x)
        #obtain concatenated filters
        xm0=self.attm0(x)
        #xm1=self.attm1(x)
        xm2=self.attm2(x)
        xm3=self.attm3(x)
        xm4=self.attm4(x)

        xc0=self.attc0(x)
        xc1=self.attc1(x)
        xc2=self.attc2(x)
        xc3=self.attc3(x)
        xc4=self.attc4(x)

        return [xm0,xm2,xm3,xm4],[xc0,xc1,xc2,xc3,xc4]

class MultiTaskNet(nn.Module):
    def __init__(self, att_module, hidden_dim=1024,num_filters=384,attention_hop=20,\
        num_classes_sp=6,num_classes_chn=5,num_classes_fan=6,num_classes_man=6,num_classes_cxn=5,\
        num_classes_emo=4):
        super(MultiTaskNet, self).__init__()
        self.hidden_dim=hidden_dim
        self.attention_hop = attention_hop
        self.num_filters=num_filters
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=(40,8))#, padding=3)

        self.conv2 = nn.Conv1d(1, num_filters, kernel_size=(40,16))#, padding=7)

        self.conv3 = nn.Conv1d(1, num_filters, kernel_size=(40,32))#, padding=15)

        self.conv4 = nn.Conv1d(1, num_filters, kernel_size=(40,64))#, padding=31)

        self.max_pool = nn.MaxPool1d(kernel_size=7, stride=7)
        self.dropout = nn.Dropout(0.2)

        self.att_sp=att_module(hidden_dim,num_filters,attention_hop,num_classes_sp)
        self.att_chn=att_module(hidden_dim,num_filters,attention_hop,num_classes_chn)
        self.att_fan=att_module(hidden_dim,num_filters,attention_hop,num_classes_fan)
        self.att_man=att_module(hidden_dim,num_filters,attention_hop,num_classes_man)
        self.att_cxn=att_module(hidden_dim,num_filters,attention_hop,num_classes_cxn)

    def forward(self, x):
        c1 = torch.flatten(F.relu(self.conv1(x)),start_dim=2)
        c2 = torch.flatten(F.relu(self.conv2(x)),start_dim=2)
        c3 = torch.flatten(F.relu(self.conv3(x)),start_dim=2)
        c4 = torch.flatten(F.relu(self.conv4(x)),start_dim=2)

        c1 = self.max_pool(c1)
        c2 = self.max_pool(c2)
        c3 = self.max_pool(c3)
        #c4 = self.max_pool(c4).squeeze_(2)
        if c4.shape[-1]>7:
            c4 = self.max_pool(c4)

        #print(c1.size(),c2.size(),c3.size(),c4.size())
        x = torch.cat((c1,c2,c3,c4),dim=2).transpose(2,1)
        x = self.dropout(x)
        #obtain concatenated filters
        x_sp=self.att_sp(x)
        x_chn=self.att_chn(x)
        x_fan=self.att_fan(x)
        x_man=self.att_man(x)
        x_cxn=self.att_cxn(x)
        return x_sp,x_chn,x_fan,x_man,x_cxn

