```python
# https://wikidocs.net/63618
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torchvision
from torchvision import transforms, datasets
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.io import read_image

import torch.nn.init
from ast import increment_lineno

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


import torch.nn as nn
import torch.nn.functional as F

import os
import pandas as pd

import cv2
```


```python
# 2. 딥러닝 모델 설계할 때 활용하는 장비 확인
if torch.cuda.is_available():
     DEVICE=torch.device('cuda')
else:
     DEVICE=torch.device('cpu')

print('Using Pytorch version:',torch.__version__,'Device:',DEVICE)
```

    Using Pytorch version: 1.11.0 Device: cuda
    


```python
BATCH_SIZE=32
learning_rate = 0.001
```


```python
root = ".././Taekwondo/DataSet/train_Image"
test_root = ".././Taekwondo/DataSet/test_Image"

trans = transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            ])

tests = transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            ])

test_dataset = torchvision.datasets.ImageFolder(root = test_root,
                                           transform = tests)

trainset = torchvision.datasets.ImageFolder(root = root,
                                           transform = trans)

train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          drop_last=True)
```


```python
# 4. 데이터 확인하기(1)
for(X_train,Y_train)in train_loader:
    print('X_train:',X_train.size(),'type:', X_train.type())
    print('Y_train:',Y_train.size(),'type:', Y_train.type())
    break
```

    X_train: torch.Size([32, 3, 224, 224]) type: torch.FloatTensor
    Y_train: torch.Size([32]) type: torch.LongTensor
    


```python
# 5. 데이터 확인하기(2)
pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)     # 여러 그래프 그리기, 첫숫자 : 행, 둘째 : 열
    plt.axis('off')           # 축없음
    plt.imshow(np.transpose(X_train[i],(1,2,0)))
    plt.title('Class: ' + str(Y_train[i].item()))
# imshow : 이미지 출력
```


    
![png](output_5_0.png)
    



```python
# 6. MLP(Multi Layer Perceptron) 모델 설계하기
class CNN(nn.Module):                 # nn.Module을 상속받는 Net클래스 생성
    def __init__(self):   
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, 
                               out_channels = 8, 
                               kernel_size = 3, 
                               padding = 1)
        # 2차원 이미지 데이터를 nn.Conv2d메서드를 이용해 Convolution연산을 하는 filter를 정의
        
        # in_channels = 3
        # 채널 수를 이미지의 채널수와 맞춰야 한다. RGB 채널은 채널수 = 3 이다.
        
        # out_channels = 8
        # Convolution연산을 진행하는 필터 개수
        # 여기서 설정해주는 Filter개수만큼 Output의 depth가 정해집니다.
        # 여기서는 depth가 8인 Feature Map이 생성
        
        # kernel_size = 3
        # Filter의 크기. 스칼라 값으로 설정하려면 가로 * 세로 크기인 Filter를 이용.
        # kernel_size = 3이면 3*3 의 필터가 이미지 위를 돌아다니면서 겹치는 영역에 대해
        # 9개의 픽셀 값과  Filter내에 있는 9개의 파라미터 값을 Convolution연산으로 진행.
        
        self.conv2 = nn.Conv2d(in_channels = 8, 
                               out_channels = 16, 
                               kernel_size = 3, 
                               padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1=nn.Linear(8*8*16,64) # (input node수, output node수)
        self.fc2=nn.Linear(64,32)   # 이전 output node수 = 다음 input node수
        self.fc3=nn.Linear(32,10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(-1, 8 * 8 * 16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)
        return x
    
```


```python
#7.Optimizer, Object Function 설정하기
model=CNN().to(DEVICE)                  # MLP모델을 기존에 선정한 'DEVICE'에 할당합니다. 
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
# Back Propagation을 이용해 파라미터를 업데이트할 때 이용하는 Optimizer를 정의합니다.
# SGD알고리즘을 이용하며 Learning Rate = 0.01, momentum=0.5로 설정
criterion=nn.CrossEntropyLoss()
# MLP모델의 output값과 계산될 Label값은 Class를 표현하는 원-핫 인코딩 값입니다.
# MLP모델의 output값과 원-핫 인코딩 값과의 Loss는 CrossEntropy를 이용해 계산하기 위해
# criterion은 nn.CrossEntropyLoss() 로 설정. 

print(model)
```

    CNN(
      (conv1): Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=1024, out_features=64, bias=True)
      (fc2): Linear(in_features=64, out_features=32, bias=True)
      (fc3): Linear(in_features=32, out_features=10, bias=True)
    )
    


```python
EPOCHS=20
```


```python
#8. MLP모델 학습을 진행하며 "학습 데이터"에 대한 모델 성능을 확인하는 함수 정의(train_loader)
def train(model, train_loader, optimizer, log_interval):
    model.train()         # 학습상태로 지정
    for batch_idx,(image, label) in enumerate(train_loader):
        image=image.to(DEVICE)
        label=label.to(DEVICE)
        optimizer.zero_grad()
        output=model(image)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval==0:
            print("Train Epoch: {}[{}/{}({:.0f}%)]\tTrain Loss: {:.6f}"
                  .format(Epoch,batch_idx * len(image),
                len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))
```


```python
#9. 학습되는 과정속에서 "검증데이터"에 대한 모델 성능을 확인하는 함수 정의(test_loader)
def evaluate(model,test_loader):
    model.eval()                  # 평가상태로 지정
    test_loss=0
    correct=0
    
    with torch.no_grad():
        for image, label in test_loader:
            image=image.to(DEVICE)              # 8과 동일
            label=label.to(DEVICE)              # 8과 동일
            output=model(image)                 # 8과 동일
            test_loss+=criterion(output,label).item()
            prediction = output.max(1,keepdim = True)[1]
            # MLP 모델의 output값은 크기가 10인 벡터값입니다. 
            # 계산된 벡터값 내 가장 큰 값인 위치에 대해 해당 위치에 대응하는 클래스로 예측했다고 판단합니다.
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            # MLP모델이 최종으로 예측한 클래스 값과 실제 레이블이 의미하는 클래스가 맞으면 correct에 더해 올바르게 예측한 횟수를 저장
                        
    test_loss /= len(test_loader.dataset)
    # 현재까지 계산된 test_loss 값을 test_loader내에 존재하는 Mini-Batch 개수(=10)만큼 나눠 평균 Loss값으로 계산.
    test_accuracy=100.*correct / len(test_loader.dataset)
    # test_loader 데이터중 얼마나 맞췄는지를 계산해 정확도를 계산합니다.
    return test_loss, test_accuracy
```


```python
#10. MLP학습을 실행하면서 Train, Test set의 Loss및 Test set Accuracy를 확인하기
for Epoch in range(1,EPOCHS+1):
    train(model,train_loader,optimizer,log_interval = 200)
    test_loss, test_accuracy=evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".
          format(Epoch,test_loss,test_accuracy))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Input In [12], in <cell line: 2>()
          1 #10. MLP학습을 실행하면서 Train, Test set의 Loss및 Test set Accuracy를 확인하기
          2 for Epoch in range(1,EPOCHS+1):
    ----> 3     train(model,train_loader,optimizer,log_interval = 200)
          4     test_loss, test_accuracy=evaluate(model, test_loader)
          5     print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".
          6           format(Epoch,test_loss,test_accuracy))
    

    Input In [10], in train(model, train_loader, optimizer, log_interval)
          7 optimizer.zero_grad()
          8 output=model(image)
    ----> 9 loss=criterion(output,label)
         10 loss.backward()
         11 optimizer.step()
    

    File C:\Anaconda3\envs\project\lib\site-packages\torch\nn\modules\module.py:1110, in Module._call_impl(self, *input, **kwargs)
       1106 # If we don't have any hooks, we want to skip the rest of the logic in
       1107 # this function, and just call forward.
       1108 if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1109         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1110     return forward_call(*input, **kwargs)
       1111 # Do not call functions when jit is used
       1112 full_backward_hooks, non_full_backward_hooks = [], []
    

    File C:\Anaconda3\envs\project\lib\site-packages\torch\nn\modules\loss.py:1163, in CrossEntropyLoss.forward(self, input, target)
       1162 def forward(self, input: Tensor, target: Tensor) -> Tensor:
    -> 1163     return F.cross_entropy(input, target, weight=self.weight,
       1164                            ignore_index=self.ignore_index, reduction=self.reduction,
       1165                            label_smoothing=self.label_smoothing)
    

    File C:\Anaconda3\envs\project\lib\site-packages\torch\nn\functional.py:2996, in cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing)
       2994 if size_average is not None or reduce is not None:
       2995     reduction = _Reduction.legacy_get_string(size_average, reduce)
    -> 2996 return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
    

    ValueError: Expected input batch_size (1568) to match target batch_size (32).



```python

```
