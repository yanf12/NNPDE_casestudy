'''
this section import iris data and visualize it

'''

import torch
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import torch.nn as nn
# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
import pandas as pd
import torch.optim as optim

# import some data to play with
iris = datasets.load_iris()
df = pd.DataFrame(iris.data[:, :2],columns=["sepal length (cm)","sepal width (cm)"])
df["type"] = iris.target
df = df.drop(df.index[df["type"]==2])
X = np.array(df.iloc[:,[0,1]])
y = np.array(df.iloc[:,-1])
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
plt.figure(2, figsize=(8, 6))
plt.clf()
# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


# pytorch version of the neural network
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(2, 6)
        self.output = nn.Linear(6, 1)

        # activation
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.input(x)

        x = self.sigmoid(x)

        x = self.output(x)
        return x

if __name__ == "__main__":

    torch.manual_seed(1024)

    model = Model()


    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)


    h = 0.01 # step size for finite difference approximation
    loss_list = []

    epoch = 150
    for n in range(epoch):
        for i in range(X.shape[0]):
            x = torch.tensor(X[i]).float()
            target_y = torch.tensor(float(y[i]))

            #loss = torch.tensor([[0.]])
            optimizer.zero_grad()
            pred_y = model.forward(x)
            cost = -(target_y*torch.log(pred_y)+(1-target_y)*torch.log(1-pred_y))
            cost.backward()
            optimizer.step()
            loss_list.append(cost.detach().numpy()[0])

plt.plot(loss_list)
plt.title("lost by SGD")
plt.show()
#%%
from torch.autograd import Variable
# draw a heatmap for visualising the decision boundary
x1min, x1max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2min, x2max = X[:, 1].min() - 1, X[:, 1].max() + 1
steps = 1000
x1_span = np.linspace(x1min, x1max, steps)
x2_span = np.linspace(x2min, x2max, steps)
xx1, xx2 = np.meshgrid(x1_span, x2_span)
color_map = plt.get_cmap('Paired')

# Make predictions across region of interest
model.eval()
labels_predicted = model(Variable(torch.from_numpy(np.c_[xx1.ravel(), xx2.ravel()]).float()))

# Plot decision boundary in region of interest
labels_predicted = [0 if value <= 0.5 else 1 for value in labels_predicted.detach().numpy()]
z = np.array(labels_predicted).reshape(xx1.shape)

# Plot decision boundary in region of interest
labels_predicted = [0 if value <= 0.5 else 1 for value in labels_predicted]
z = np.array(labels_predicted).reshape(xx1.shape)

fig, ax = plt.subplots()
ax.contourf(xx1, xx2, z, cmap=color_map, alpha=0.5)

# Get predicted labels on training data and plot
#train_labels_predicted = model(X)
ax.scatter(X[:, 0], X[:, 1], c=y.reshape(y.size), cmap=color_map, lw=0,label="true data points")
plt.legend()
plt.show()
