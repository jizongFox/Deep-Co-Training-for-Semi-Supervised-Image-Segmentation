import torch
import torch.nn.functional as F
import numpy as np

def sigmoid(x):
    if isinstance(x,list):
        x = np.array(x)
    return 1 / (1 + np.exp(-x))
w = torch.ones(1,1,requires_grad=True)
net = lambda x: F.sigmoid(w*x)
X= torch.from_numpy(np.linspace(-10,10,1000))
Y = -F.sigmoid(X)*F.sigmoid(X).log()
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(X.data.numpy(),Y.data.numpy())
plt.title('KL_loss')
plt.show(block=False)
G = []
gradients = []
losses = []
eps = 1e-2

## using backward method

for i in X:
    x = torch.Tensor([i])
    p = net(x)
    # loss1 = (p.detach()*p.log().detach()- p.detach()*(p).log()).mean()
    loss1 = (p.detach()-p)**2
    loss1.backward()
    losses.append(loss1.item())
    G.append(w.grad.data.numpy()[0][0])
    w.grad.zero_()
plt.figure(2)
plt.plot(X.tolist(),G)
plt.title('Gradient calculated by backward')
plt.show(block=False)
# plt.pause(20)


## using handcalculate
for i in X:

    x0 = torch.Tensor([i])
    p0 = net(x0)
    x1 = torch.Tensor([i+eps])
    p1 = net(x1)

    diff = - p1*p1.log() -(- p0*p0.log())
    # diff = torch.abs(p1-p0)
    g = diff/eps
    gradients.append(g)

plt.figure(3)
plt.plot(X.tolist(),gradients)
plt.title('hand_made gradient')
plt.show()
# #
