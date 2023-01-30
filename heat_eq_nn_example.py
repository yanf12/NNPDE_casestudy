import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
'''
this script solves ut = uxx  on [0,1] with u(0,t) = 0, u(1,t) =0
u(x,0) = f(x);
here we take f(x) = 6 sin(pi*x)
'''

def f(x):
    return 6*np.sin(np.pi*x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(2, 32) # there are two features as input: x and t

        self.hidden_1 = nn.Linear(32, 32)
        self.hidden_2= nn.Linear(32, 32)

        self.output = nn.Linear(32,1)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.input(x)

        x = self.hidden_1(x)
        x = self.sigmoid(x)
        x = self.hidden_2(x)
        x = self.sigmoid(x)

        x = self.output(x)
        return x

if __name__ == "__main__":

    torch.manual_seed(1024)

    model = Model()


    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_iteration = 2000
    h = 0.01 # step size for finite difference approximation
    dt = 0.01 # time step
    loss_list = []

    for i in range(num_iteration):
        print(i)


        for t in np.linspace(0,1,10):
            for x in np.linspace(0,1,10):
                loss = torch.tensor([[0.]])
                optimizer.zero_grad()
                x= float(x)
                t = float(t)
                y_pred = model.forward(torch.tensor([x,t]))
                y_minus_h = model.forward(torch.tensor([x-h,t]))
                y_plus_h = model.forward(torch.tensor([x+h,t]))
                y_minus_dt = model.forward(torch.tensor([x,t-dt]))
                y_plus_dt = model.forward(torch.tensor([x,t+dt]))
                y_2nd_deri_space= (y_plus_h-2*y_pred+y_minus_h)/(h**2)
                y_1st_deri_time=(y_plus_dt-y_minus_dt)/(2*dt)
                #y_1st_deri = (y_plus_h-y_minus_h)/(2*h)

                if x==0 or x ==1:
                    loss += F.mse_loss(y_pred, torch.tensor([float(0)]))
                elif t ==0 :
                    loss += F.mse_loss(y_pred, torch.tensor([float(f(x))]))
                else:
                    loss += F.mse_loss(y_1st_deri_time-y_2nd_deri_space,torch.tensor([float(0)]))

                loss.backward()
                optimizer.step()
                loss_list.append(loss.detach().numpy()[0][0])


    # see the model prediction after training
    pred_list = []
    exact_list = []
    t=0.0
    for x in np.linspace(0,1,10):
        pred_list.append(model.forward(torch.tensor([[float(x),t]])).detach().numpy()[0][0])
        exact_list.append(6*np.sin(np.pi*x)*np.exp(-1*(np.pi**2)*t))
    plt.figure()
    plt.plot(np.linspace(0,1,10),pred_list,label = "NN Result")
    plt.plot(np.linspace(0,1,10),exact_list,label ="Exact Result")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(loss_list)
    plt.title("loss:SGD")
    plt.show()




