#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:37:23 2018

@author: Dominik Schmidt
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


#comments GK:
    #some comments:
      #torch works with tensors (multidimensional arrays)
      #torch tensors have very similar functions as numpy objects
      #PyTorch allows flexible code and debugging
      #autograd (automatic gradient) keeps track of all gradients for you
      #PyTorch builds the network dynamically, allows you to change it also dynamically

#%%
T = 10          # Length of time series
Tinp = 3        # Time at which the input is presented
Tout = [7]      # Time points at which the output is required
N = 10           # Number of neurons
Ntrial = 6000   # Number of trials
alpha = 0.1     # learning rate
batchsize = 5   # how many trials per training step?

# Create 1 layer RNN with N neurons
# Inputs to RNN: 
# No. of input units, No. of total units, No. of layers, nonlinearity
net = nn.RNN(N, N, 1, nonlinearity='tanh')
#net = nn.RNN(N, N, 1, nonlinearity='relu')

#%%
# Define the inputs for one trial:
# Inputs to the RNN need to have 3 dimensions, where the first dimension is
# the length of the time series, the second is the trial number and the third
# is the number of the unit that receives the input. Here, we define the input
# for one single trial, so we set the 2nd dimension to 1
input1 = np.zeros([T,1,N])
input2 = np.zeros([T,1,N])
input1[Tinp,0,0] = 1   # Provide input to neuron 0
input2[Tinp,0,1] = 1   # Provide input to neuron 1

# Define the targets for a single trial
target1 = np.zeros([T,1,N])
target2 = np.zeros([T,1,N])
for t in Tout:
    target1[Tout,0,3] = 1
    target2[Tout,0,2] = 1
inputs  = {0: input1, 1: input2}
targets = {0: target1, 1: target2}


# We want to record losses for plotting
losses = np.zeros(Ntrial)   
# Define the loss function. We want to use mean square error function
criterion = nn.MSELoss()
for i in range(Ntrial):
    # First, stitch multiple input-target pairs together to one batch
    # Notice that the trial number within one batch is given by the second
    # dimension of the input/target tensor!
    inpt    = torch.zeros(T,batchsize,N, dtype=torch.float)
    target  = torch.zeros(T,batchsize,N, dtype=torch.float)
    for n in range(batchsize):
        # Draw randomly from either trial type 1 or trial type 2
        trial_type      = np.random.randint(0,2)
        inpt[:,n,:]     = torch.tensor(inputs[trial_type], dtype=torch.float).squeeze()
        target[:,n,:]   = torch.tensor(targets[trial_type], dtype=torch.float).squeeze()
        
    # To propagate input through the network, we simply call the network with
    # the input as argument. Output is the whole time series for all trials in
    # the batch
    [outp, _] = net(inpt)
    
    # Calculate MSE between output and target. Here we specifically select
    # those units (2 and 3) and time points (Tout) that have target outputs.
    loss = criterion(outp[Tout,:,2:4], target[Tout,:,2:4])
    losses[i] = loss
    
    # We need to reset the gradients from previous step to zero:
    net.zero_grad()
    
    # This function backpropagates the loss through the network. PyTorch takes 
    # care of calculating the gradients that are created by the backpropagation
    loss.backward()
    
    # Use the optimizer on the parameters, with learning rate alpha
    # We use stochastic gradient descent, but you can change it if you want:
    optimizer = optim.SGD(net.parameters(), lr=alpha)
    # Finally, do one gradient descent step:
    optimizer.step()


# Now we plot the results. For that, we propagate both input types through the
# network and plot the resulting time series
plt.subplot(3,1,1)
plt.plot(losses)
inputs_dict  = {0: input1, 1: input2}
targets_dict = {0: target1, 1: target2}
for i in range(2):
    inpt = torch.tensor(inputs_dict[i], dtype=torch.float)
    
    [outp, _] = net(inpt)
    outp = outp.detach().numpy()
    x3 = np.squeeze(outp[:,0,2])
    x4 = np.squeeze(outp[:,0,3])
    plt.subplot(3,1,i+2)
    plt.plot(range(T), x3)
    plt.plot(range(T), np.zeros(T), 'k--', alpha=0.3)
    plt.plot(range(T), np.ones(T), 'k--', alpha=0.3)
    plt.plot(range(T), x4)
    plt.plot(range(T), np.zeros(T), 'k--', alpha=0.3)
    plt.plot(range(T), np.ones(T), 'k--', alpha=0.3)
