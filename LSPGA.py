import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class LSPGA(object):
    def __init__(self,model,epsilon,k,delta,xi,step,criterion,encoder):
        self.model = model
        self.epsilon = epsilon        # l-inf norm bound of image
        self.k=k                      # k-level quantization
        self.delta = delta            # annealing factor
        self.xi = xi                  # step-size of attack
        self.step = step              # attack steps
        self.criterion = criterion    # loss func
        self.encoder = encoder        # temp encoder

    def getMask(self,x):
        n,w,h = x.shape
        mask = np.zeros((n,self.k,w,h))
        low = x - self.epsilon
        low[low < 0] = 0
        high = x + self.epsilon
        high[high > 1] = 1
        for i in range(self.k+1):
            interimg = (i*1./self.k)*low + (1-i*1./self.k)*high
            mask+=self.encoder.onehotencoding(interimg)
        mask[mask>1] = 1
        return mask

    def attackthreechannel(self, data, target):
        target = Variable(target.cuda())
        datanumpy = data.numpy()
        channel0, channel1, channel2 = (datanumpy[:, i, :, :] for i in range(3))
        mask0, mask1, mask2 = (self.getMask(channel) for channel in [channel0, channel1, channel2])
        u0, u1, u2 = (np.random.random(mask.shape) - (1 - mask) * 1e10 for mask in [mask0, mask1, mask2])
        T = 1.0
        u0, u1, u2 = (Variable(torch.Tensor(u).cuda(), requires_grad=True) for u in [u0, u1, u2])
        z0, z1, z2 = (F.softmax(u / T, dim=1) for u in [u0, u1, u2])
        z0, z1, z2 = (torch.cumsum(z, dim=1) for z in [z0, z1, z2])
        for t in range(self.step):
            out = self.model(z0, z1, z2)
            loss = self.criterion(out, target)
            for u in [u0, u1, u2]:
                if u.grad != None:
                    u.grad.data._zero()
            loss.backward()
            grad0, grad1, grad2 = (u.grad for u in [u0, u1, u2])
            u0, u1, u2 = (self.xi * torch.sign(grad) + u for (grad, u) in zip([grad0, grad1, grad2], [u0, u1,u2]))
            u0, u1, u2 = (Variable(u.data, requires_grad=True) for u in [u0, u1, u2])
            z0, z1, z2 = (F.softmax(u / T, dim=1) for u in [u0, u1, u2])
            z0, z1, z2 = (torch.cumsum(z, dim=1) for z in [z0, z1, z2])
            T = T * self.delta
        c0, c1, c2 = (np.argmax(u.data.cpu().numpy(), axis=1) for u in [u0, u1, u2])
        them0, them1, them2 = (self.encoder.tempencoding(c) for c in [c0, c1, c2])
        return them0, them1, them2

    def attackthreechannel_train(self, data, target):
        target = Variable(target.cuda())
        datanumpy = data.numpy()
        channel0, channel1, channel2 = (datanumpy[:, i, :, :] for i in range(3))
        mask0, mask1, mask2 = (self.getMask(channel) for channel in [channel0, channel1, channel2])
        u0, u1, u2 = (np.random.random(mask.shape) - (1 - mask) * 1e10 for mask in [mask0, mask1, mask2])
        T = 1.0
        u0, u1, u2 = (Variable(torch.Tensor(u).cuda(), requires_grad=True) for u in [u0, u1, u2])
        z0, z1, z2 = (F.softmax(u / T, dim=1) for u in [u0, u1, u2])
        z0, z1, z2 = (torch.cumsum(z, dim=1) for z in [z0, z1, z2])
        for t in range(self.step):
            out = self.model(z0, z1, z2)
            loss = self.criterion(out, target)
            for u in [u0, u1, u2]:
                if u.grad != None:
                    u.grad.data._zero()
            loss.backward()
            grad0, grad1, grad2 = (u.grad for u in [u0, u1, u2])
            u0, u1, u2 = (self.xi * torch.sign(grad) + u for (grad, u) in zip([grad0, grad1, grad2], [u0, u1,u2]))
            u0, u1, u2 = (Variable(u.data, requires_grad=True) for u in [u0, u1, u2])
            z0, z1, z2 = (F.softmax(u / T, dim=1) for u in [u0, u1, u2])
            z0, z1, z2 = (torch.cumsum(z, dim=1) for z in [z0, z1, z2])
            T = T * self.delta
        them0, them1, them2 = (z.data.cpu().numpy() for z in [z0, z1, z2])
        return them0, them1, them2

    def attackonechannel(self, data, target):
        target = Variable(target.cuda())
        datanumpy = data.numpy()
        data0 = datanumpy[:, 0, :, :]
        mask = self.getMask(data0)
        u = np.random.random(mask.shape) - (1 - mask) * 1e10
        T = 1.0
        u = Variable(torch.Tensor(u).cuda(), requires_grad=True)
        z = F.softmax(u / T, dim=1)
        z = torch.cumsum(z, dim=1)
        for t in range(self.step):
            out = self.model(z)
            loss = self.criterion(out, target)
            if u.grad != None:
                u.grad.data._zero()
            loss.backward()
            grad = u.grad
            u = self.xi * torch.sign(grad) + u
            u = Variable(u.data, requires_grad=True)
            z = F.softmax(u / T, dim=1)
            z = torch.cumsum(z, dim=1)
            T = T * self.delta
        attackimg = np.argmax(u.data.cpu().numpy(), axis=1)
        themattackimg = self.encoder.tempencoding(attackimg)
        return themattackimg
