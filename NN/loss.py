import torch
import torch.nn as nn

class FmeasureLoss(nn.Module):
    def __init__(self, num_classes=4, beta=1):
        super(FmeasureLoss, self).__init__()
        self.EPS= torch.tensor(1.0e-8).cuda()
        self.n_class = num_classes
        self.softmax= nn.Softmax(dim=1)
        self.beta = torch.tensor(beta**2).cuda()

    def forward(self, y, lab):
        N = y.size(0)
        q = self.softmax(y)

        qp = q / torch.max(q, 1, True)[0]

        y_onehot = torch.FloatTensor(y.size(0), self.n_class).cuda()
        y_onehot.zero_()
        lab = lab.view(-1,1)
        y_onehot = y_onehot.scatter_(1, lab, 1)

        Nk = torch.sum(y_onehot, 0, True)
        # Only kept those value for k
        # tp(k) true positive value for all the softmax activations
        num = torch.matmul(torch.t(y_onehot), torch.sum(y_onehot * qp, 1, True))*(1+self.beta)
        #Each column corresponds to a class k tp(k)+fp(k)
        den = torch.t(self.beta* Nk + torch.sum(qp, 0, True) + self.EPS)

        loss = - torch.mean(num/den)
        #loss = - torch.sum(num*torch.t(Nk)/den/N)
        return loss
