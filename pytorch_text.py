import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2

external_grad = torch.tensor([1., 1.])

Q.retain_grad()
Z=Q**2
Z.backward(gradient = external_grad)


