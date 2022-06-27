import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import random_split, DataLoader

# Model Definition
model = nn.Sequential(nn.Linear(28*28, 64),
                      nn.ReLU(),
                      nn.Linear(64,64),
                      nn.ReLU(),
                      nn.Linear(64, 10)
                      ).cuda()

# Optimiser Definition
optimiser = optim.SGD(model.parameters(), lr = 1e-2)

# Loss Function Definition
loss = nn.CrossEntropyLoss()

#Train, Val split
train_data = datasets.MNIST('data',train = True, download = True, transform = transforms.ToTensor())
train, val = random_split(train_data, [50000,10000])
train_loader = DataLoader(train,batch_size = 25)
val_loader = DataLoader(val, batch_size = 25)

# Training & Validation Loops
nb_epochs = 10
for epoch in range(nb_epochs):
    losses = list()
    accuracies = list()
    for batch in train_loader:
        x, y = batch
        
        # x: b*1*28*28 converts a 28*28 matrix into a vector input
        b = x.size(0)
        x = x.view(b, -1).cuda()
        
        # 1 forward
        l = model(x) # l: logit
        
        # 2 Compute the objective
        J = loss(l,y.cuda())
        
        
        # 3 cleaning the gradients
        model.zero_grad()
        
        # 4 Compute the partial derivatives of J wrt paraameters
        J.backward()
        
        # 5 Step in the opposite direction of the gradient
        optimiser.step()   
        
        losses.append(J.item())
        
        accuracies.append(y.eq(l.detach().argmax(dim = 1).cpu()).float().mean())
        
    print(f'Epoch {epoch + 1}', end =', ')
    print(f'Training loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'training accuracy: {torch.tensor(accuracies).mean():.2f}')
    
    losses = list()
    accuracies = list()
    for batch in val_loader:
        x, y = batch
        
        # x: b*1*28*28 converts a 28*28 matrix into a vector input
        b = x.size(0)
        x = x.view(b, -1).cuda()
        
        # 1 forward
        with torch.no_grad():
            l = model(x) # l: logit
        
        # 2 Compute the objective
        J = loss(l,y.cuda())
        
        losses.append(J.item())
        
        accuracies.append(y.eq(l.detach().argmax(dim = 1).cpu()).float().mean())
        
    print(f'Epoch {epoch + 1}', end =', ')
    print(f'Validation loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'Validation accuracy: {torch.tensor(accuracies).mean():.2f}')