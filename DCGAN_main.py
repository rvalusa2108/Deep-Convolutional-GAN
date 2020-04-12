# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 01:29:03 2020

@author: Raghu
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from collections import OrderedDict

# Setting some hyperparameters
batchSize = 64 # We set the size of the batch.
imageSize = 64 # We set the size of the generated images (64x64).

# Creating the transformations
#transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root='./cifar10data', 
                       download=False, 
                       transform=transform) # We downloaded the training set in the ./cifar10data folder and we apply the previous transformations on each image. We already downloaded the cifar10 data.
dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                         batch_size=batchSize, 
                                         shuffle=True, 
                                         num_workers=4) # We use dataLoader to get the images of the training set batch by batch.

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the generator
class G(nn.Module): # We introduce a class to define the generator.
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(OrderedDict({
          'conv1': nn.ConvTranspose2d(in_channels=100, 
                                       out_channels=512, 
                                       kernel_size=4, 
                                       stride=1, #default
                                       padding=0, #default
                                       output_padding=0, #default
                                       groups=1, #default
                                       bias=True, #default, try False later
                                       dilation=1, #default
                                       padding_mode='zeros'), #default
          'batchnorm1': nn.BatchNorm2d(num_features=512, 
                                        eps=1e-5, #default
                                        momentum=0.1, #default
                                        affine=True, #default
                                        track_running_stats=True), #default
          'relu1': nn.ReLU(inplace=True), #default False
          ##
          'conv2': nn.ConvTranspose2d(in_channels=512, 
                                       out_channels=256, 
                                       kernel_size=4, 
                                       stride=2, #default 1
                                       padding=1, #default 0
                                       output_padding=0, #default
                                       groups=1, #default
                                       bias=True, #default, try False later
                                       dilation=1, #default
                                       padding_mode='zeros'), #default
          'batchnorm2': nn.BatchNorm2d(num_features=256, 
                                        eps=1e-5, #default
                                        momentum=0.1, #default
                                        affine=True, #default
                                        track_running_stats=True), #default
          'relu2': nn.ReLU(inplace=True), #default False
          ##
          'conv3': nn.ConvTranspose2d(in_channels=256, 
                                       out_channels=128, 
                                       kernel_size=4, 
                                       stride=2, #default 1
                                       padding=1, #default 0
                                       output_padding=0, #default
                                       groups=1, #default
                                       bias=True, #default, try False later
                                       dilation=1, #default
                                       padding_mode='zeros'), #default
          'batchnorm3': nn.BatchNorm2d(num_features=128, 
                                        eps=1e-5, #default
                                        momentum=0.1, #default
                                        affine=True, #default
                                        track_running_stats=True), #default
          'relu3': nn.ReLU(inplace=True), #default False
          ##
          'conv4': nn.ConvTranspose2d(in_channels=128, 
                                       out_channels=64, 
                                       kernel_size=4, 
                                       stride=2, #default 1
                                       padding=1, #default 0
                                       output_padding=0, #default
                                       groups=1, #default
                                       bias=True, #default, try False later
                                       dilation=1, #default
                                       padding_mode='zeros'), #default
          'batchnorm4': nn.BatchNorm2d(num_features=64, 
                                        eps=1e-5, #default
                                        momentum=0.1, #default
                                        affine=True, #default
                                        track_running_stats=True), #default
          'relu4': nn.ReLU(inplace=True), #default False
          ##
          'conv5': nn.ConvTranspose2d(in_channels=64, 
                                       out_channels=3, 
                                       kernel_size=4, 
                                       stride=2, #default 1
                                       padding=1, #default 0
                                       output_padding=0, #default
                                       groups=1, #default
                                       bias=True, #default, try False later
                                       dilation=1, #default
                                       padding_mode='zeros'), #default
          'relu5': nn.Tanh() #default False
        }))


    def forward(self, input):
        output = self.main(input)
        return output

# Creating the generator
netG = G()
netG.apply(weights_init)


class D(nn.Module):
    def __init__(self,):
        super(D, self).__init__()
        
        self.main = nn.Sequential(
                            nn.Conv2d(in_channels=3, 
                                      out_channels=64, 
                                      kernel_size=4, 
                                      stride=2, #default=1
                                      padding=1, #default=0
                                      dilation=1, 
                                      groups=1, 
                                      bias=True, 
                                      padding_mode='zeros'),
                            # nn.BatchNorm2d(num_features=64, #I added this extra
                            #                 eps=1e-5, #default
                            #                 momentum=0.1, #default
                            #                 affine=True, #default
                            #                 track_running_stats=True),
                            nn.LeakyReLU(negative_slope=0.2, #default=1e-2 = 0.01
                                         inplace=True),
                            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
                            nn.BatchNorm2d(128),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(128, 256, 4, 2, 1, bias=True),
                            nn.BatchNorm2d(256),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(256, 512, 4, 2, 1, bias=True),
                            nn.BatchNorm2d(512),
                            nn.LeakyReLU(0.2, inplace=True),    
                            nn.Conv2d(512, 1, 4, 1, 0, bias=True),
                            nn.Sigmoid())


    def forward(self, input):
        output = self.main(input)
        return output.view(-1)



# Creating the discriminator
netD = D()
netD.apply(weights_init) #initialozing the weights of Dicriminator NN by applying the weights defined earlier.


#Training the DCGANs
criterion = nn.BCELoss() #Loss function - Binary Cross Entrophy Loss(BCELoss)
optimizerD = optim.Adam(params=netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(params=netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(25): #iterates from 0 through 24 ==> total 25 iterations
    for i, data in enumerate(dataloader, 0):
        # 1st Step: Updating the weights of the neural network of the discriminator
        
        netD.zero_grad() # initializing the gradients of Driciminator neutral network to 0 for the initial weights defined above.
        
        # Training the discriminator with a real image of the dataset
        real, _ = data # minibatch of images gives two outputs, we are interested in first output value and assigning that variable named - real, second _ mean, we are ignoring the second output.
        input = Variable(real) #converting the real variable to torch tensor variable using Variable method
        target = torch.ones(input.size()[0]) #target for training the Discriminator with real images from minibatch is 1. So using torch.ones to create the torch variable of ones of size equal to input
        output = netD(input) # This will call the special function __call__ present in Module class, which is a super class inherited in D class. This inputs the real images tensor variable into forward function
        errD_real = criterion(output, target) #Calculate the Binary Cross Entropy loss or error

        # Training the discriminator with a fake image generated by the generator
        # noise = Variable(torch.randn(size=input.size()[0], 
        #                              out=100, 
        #                              layout=(1, 1))) #random vector of tensor Variable containing 64 minibatches each with 100 random elements with each element having shape as (1,1)
        noise = Variable(torch.randn(input.size()[0], 
                                     100, 
                                     1, 1)) #random vector of tensor Variable containing 64 minibatches each with 100 random elements with each element having shape as (1,1)        
        
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)
        
        # Backpropagating the total error into Discriminator
        errD = errD_real + errD_fake #Calculate the total error 
        errD.backward() #back propagate the error into Discriminator to updates the weights so that loss fn is reduced.
        optimizerD.step()
        
        # 2nd Step: Updating the weights of the neural network of the generator
        # The loss error between the prediction error of the discriminator whether the images generated by the generator will be accepted yes or no and the target which will be equal to one.
        netG.zero_grad() #Initializing the gradient of Generator to zeros, with respective to the weights applied earlier.
        target = Variable(torch.ones(input.size()[0])) #In here the target variable in a tensor Variable of all ones with the size same as input size of minibatch of images i.e., 64
        output = netD(fake) #Output is the value generated by the Decriminator when fake images(output of Generator is passed as input to Discriminator) which is in between 0 - 1
        errG = criterion(output, target) #calculating the BCE loss between target and output
        errG.backward() #back propagate the error into Generator to updates the weights so that loss fn is reduced.
        optimizerG.step() #apply the optimizer to get the best design.
        
        # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps
        
        # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0])) # We print les losses of the discriminator (Loss_D) and the generator (Loss_G).
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data, errG.data)) # We print les losses of the discriminator (Loss_D) and the generator (Loss_G).
        if i % 100 == 0: # Every 100 steps:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True) # We save the real images of the minibatch.
            fake = netG(noise) # We get our fake generated images.
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True) # We also save the fake generated images of the minibatch.
            
            
            
            
            
            
            
            
