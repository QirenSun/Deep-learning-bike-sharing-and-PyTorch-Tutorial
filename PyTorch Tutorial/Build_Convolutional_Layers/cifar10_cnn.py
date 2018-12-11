#Load the data
from torchvision import datasets
import torchvision .transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
numworker=0
batch_size=20
valid_size=0.2

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_data = datasets.CIFAR100('data',train=True,download=True,transform=transform)
test_data = datasets.CIFAR100('data',train=False,download=True,transform=transform)

len_train= len(train_data)
indices = list(range(len_train))
np.random.shuffle(indices)
split=int(np.floor(valid_size*len_train))
train_idx,valid_idx= indices[split:],indices[:split]

train_sampler= SubsetRandomSampler(train_idx)
vaild_sampler= SubsetRandomSampler(valid_idx)

train_loader= torch.utils.data.Dataloader(train_data,batch_size=batch_size,sampler=train_sampler,
                                          num_workers=numworker)
test_loader= torch.utils.data.Dataloader(test_data,batch_size=batch_size,sampler,
                                          num_workers=numworker)
valid_loader= torch.utils.data.Dataloader(valid_data,batch_size=batch_size,sampler=valid_sampler,
                                          num_workers=numworker)

classes=['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

#Visualize a Batch of Training Data
import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])




