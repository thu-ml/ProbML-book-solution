'''
这里给出使用卷积神经网络进行手写数字识别的参考实现，鼓励读者尝试不同的网络结构、优化算法以及模型参数
'''

import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# torch.manual_seed(0) 
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

EPOCH = 20
BATCH_SIZE = 128
LR = 0.001
DOWNLOAD = False

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=data_tf,                              
    download=DOWNLOAD,
)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=data_tf)

print(train_data.data.size())
print(train_data.targets.size())      

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


model = CNN()
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()                    

loss_trace = list()

for epoch in range(EPOCH):
    losses_per_epoch = list()
    for step, (b_x, b_y) in enumerate(train_loader): 

        output = model(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print('Epoch:{} loss:{}'.format(epoch, loss.item()))

        losses_per_epoch.append(loss.item())
    loss_trace.append(np.mean(losses_per_epoch))

model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(
    correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
