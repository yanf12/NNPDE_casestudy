import matplotlib.pyplot as plt
import numpy as np
import torch
def plot_nn_ode_result(model,num_epochs,loss_list):
    # see the model prediction after training
    pred_list = []
    exact_list = []
    for x in np.linspace(0,1,100):
        pred_list.append(model.forward(torch.tensor([[float(x)]])).detach().numpy()[0][0])
        exact_list.append(1+x*(1-x))
    plt.figure()
    plt.plot(np.linspace(0,1,100),pred_list,label = "NN Result")
    plt.plot(np.linspace(0,1,100),exact_list,label ="Exact Result")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(range(num_epochs),loss_list)
    plt.title("loss against number of epochs")
    plt.grid()
    plt.show()

    

    plt.figure()
    plt.plot(np.log(range(num_epochs)),np.log(np.array(loss_list)))
    plt.title("log(loss) against log(number of epochs)")
    plt.xlabel("log(number of epochs)")
    plt.ylabel("log(loss)")
    plt.grid()
    plt.show()