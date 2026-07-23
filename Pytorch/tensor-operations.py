import torch 

a = torch.tensor([3, 2, 1])
b = torch.tensor([4, 3, 5])

print(a+b)
# tensor([7, 5, 6])

print(a-b)
# tensor([-1, -1, -4])

print(a*b)
# tensor([12,  6,  5])

print(a/b)
# tensor([0.7500, 0.6667, 0.2000])

print(a**2)

# Matrix multiplication
x = torch.rand(2,3)
y = torch.rand(3,4)

z = torch.matmul(x,y)
print("Matrix multiplication of x and y:", z)
# Matrix multiplication of x and y: tensor([[1.6205, 0.5827, 0.7643, 1.1262],
#         [1.2254, 0.4069, 0.4526, 0.8824]])


p = torch.arange(12)
print(p[0])
# tensor(0)
print(p[5])
# tensor(5)
print(p[2:8])
# tensor([2, 3, 4, 5, 6, 7])
print(p[2:8:2])
# tensor([2, 4, 6])
print(p[..., 3])
# tensor([ 3,  7, 11])
print(p[2:])
# tensor([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11])