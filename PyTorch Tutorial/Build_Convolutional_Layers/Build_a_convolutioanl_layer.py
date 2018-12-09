#Build a convolutioanl layer
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
#%matplotlib inline
image_path=r'C:\Users\Administrator\Desktop\deep learning\deep-learning-v2-pytorch-master\convolutional-neural-networks\conv-visualization\1.jpg'

image1 = cv2.imread(image_path)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

image1= image1.astype('float32')/255
plt.imshow(image1,cmap='gray')
plt.show() 

# define four filters
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])
# For an example, print out the values of filter 1
print('Filter 1: \n', filter_1)

# visualize all four filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y]<0 else 'black')
            
class Net(nn.Module):
    def __init__(self,weight):
        super(Net,self).__init__()
        k_height,k_width=weight.shape[2:]
        self.conv=nn.Conv2d(1,4,kernel_size=(k_height,k_width),bias=False)
        self.conv.weight = torch.nn.Parameter(weight)
        
    def forward(self,x):
        conv_x=self.conv(x)
        activated_x=F.relu(conv_x)
        
        return conv_x,activated_x

weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)
print(model)
# helper function for visualizing the output of a given layer
def viz_layer(layer, n_filters=4):
    fig=plt.figure(figsize=(20,20))
    
    for i in range(n_filters):
        ax= fig.add_subplot(1,n_filters,i+1,xticks=[],yticks=[])
        #grap layer output
        ax.imshow(np.squeeze(layer[0,i].data.numpy()),cmap='gray')
        ax.set_title('Output %s'%str(i+1))
# visualize all filters        
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    
image_tensor= torch.from_numpy(image1).unsqueeze(0).unsqueeze(1)
conv_layer,activated_layer= model(image_tensor)
viz_layer(conv_layer)
viz_layer(activated_layer)

    










