import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
'''
this script solves y'' = f(x) on [0,1] with y(0) = 0 
'''

def f(x):
    return -2
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(1, 6)

        self.hidden_1 = nn.Linear(32, 32)
        self.hidden_2= nn.Linear(32, 32)

        self.output = nn.Linear(6,1)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.input(x)
        x = self.sigmoid(x)

        #x = self.hidden_1(x)
        #x = self.sigmoid(x)
        #x = self.hidden_2(x)
        #x = self.sigmoid(x)

        x = self.output(x)
        return x

if __name__ == "__main__":

    torch.manual_seed(1024)

    model = Model()


    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_iteration = 1000*2
    h = 0.001 # step size for finite difference approximation
    loss_list = []

    for i in range(num_iteration):
        loss = torch.tensor([[0.]])
        optimizer.zero_grad()
        for x in np.linspace(0,1,10):
            x= float(x)
            y_pred = model.forward(torch.tensor([[x]]))
            y_minus_h = model.forward(torch.tensor([[x-h]]))
            y_plus_h = model.forward(torch.tensor([[x+h]]))
            y_2nd_deri= (y_plus_h-2*y_pred+y_minus_h)/(h**2)
            y_1st_deri = (y_plus_h-y_minus_h)/(2*h)

            if x==0:
                loss += 2*F.mse_loss(y_pred, torch.tensor([[float(1)]]))
            elif x==1:
                loss+=2*F.mse_loss(y_pred, torch.tensor([[float(1)]]))
            else:
                loss += F.mse_loss(y_2nd_deri, torch.tensor([[float(f(x))]]))

        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().numpy()[0][0])


    # see the model prediction after training
    pred_list = []
    exact_list = []
    for x in np.linspace(0,1,20):
        pred_list.append(model.forward(torch.tensor([[float(x)]])).detach().numpy()[0][0])
        exact_list.append(1+x*(1-x))
    plt.figure()
    plt.plot(np.linspace(0,1,20),pred_list,label = "NN Result")
    plt.plot(np.linspace(0,1,20),exact_list,label ="Exact Result")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(range(num_iteration),loss_list)
    plt.title("loss against number of trainings")
    plt.show()




