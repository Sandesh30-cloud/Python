## 1. What is PyTorch?

**Answer:**

PyTorch is an open-source deep learning framework developed by Meta (Facebook).

It is mainly used for:
- Building Neural Networks
- Deep Learning
- Computer Vision
- Natural Language Processing (NLP)
- Large Language Models (LLMs)

Features:
- Dynamic Computation Graph
- Automatic Differentiation (Autograd)
- GPU Support (CUDA)
- Easy Debugging

---

## 2. What is a Tensor?

**Answer:**

A Tensor is the basic data structure in PyTorch.

Think of it as a NumPy array with:
- GPU support
- Automatic gradient calculation

Examples:

0D → Scalar

1D → Vector

2D → Matrix

3D+ → Tensor

Example:

```python
x = torch.tensor([[1,2],[3,4]])
```

---

## 3. Difference between Tensor and NumPy Array?

| Tensor | NumPy |
|---------|--------|
| GPU Support | CPU Only |
| Autograd | No Autograd |
| Used in Deep Learning | Used in Scientific Computing |

---

## 4. What is a Neural Network?

**Answer:**

A Neural Network is a machine learning model inspired by the human brain.

It consists of:
- Input Layer
- Hidden Layer(s)
- Output Layer

The network learns by updating weights and biases to minimize prediction error.

---

## 5. What is a Neuron?

**Answer:**

A neuron is the smallest unit of a neural network.

Formula:

Output = Activation(WX + B)

Where:
- W = Weight
- X = Input
- B = Bias

---

## 6. What are Weights?

**Answer:**

Weights determine the importance of each input feature.

Example:

If predicting house price:

Area Weight = 0.8

Bedrooms Weight = 0.2

Area has more influence.

Weights are updated during training.

---

## 7. What is Bias?

**Answer:**

Bias is an additional parameter added to the weighted sum.

Formula:

y = wx + b

Bias helps the model fit the data better.

---

## 8. What is an Activation Function?

**Answer:**

Activation functions introduce non-linearity.

Without them, multiple neural network layers behave like one linear layer.

Common Activation Functions:
- ReLU
- Sigmoid
- Tanh
- Softmax

---

## 9. Why is ReLU mostly used?

**Answer:**

Formula:

ReLU(x) = max(0, x)

Advantages:
- Fast
- Simple
- Helps reduce vanishing gradients
- Most common hidden-layer activation

---

## 10. What is Forward Propagation?

**Answer:**

Forward propagation is the process of passing input through the network to generate predictions.

Flow:

Input
↓
Hidden Layer
↓
Output

---

## 11. What is Backpropagation?

**Answer:**

Backpropagation updates the weights using the prediction error.

Steps:
1. Make prediction
2. Calculate loss
3. Compute gradients
4. Update weights

Purpose:
Reduce prediction error.

---

## 12. What is a Loss Function?

**Answer:**

A loss function measures how wrong the model's predictions are.

Lower loss = Better model.

Examples:

Regression → Mean Squared Error (MSE)

Binary Classification → Binary Cross Entropy

Multi-Class Classification → Cross Entropy

---

## 13. What is Gradient Descent?

**Answer:**

Gradient Descent is an optimization algorithm that updates model weights to reduce the loss.

Weight Update:

New Weight = Old Weight − Learning Rate × Gradient

---

## 14. What is an Optimizer?

**Answer:**

An optimizer updates the model weights after gradients are calculated.

Common Optimizers:
- SGD
- Adam
- AdamW
- RMSProp

Adam is the most commonly used.

---

## 15. Explain the Training Loop.

**Answer:**

The training loop consists of:

1. Forward Pass
2. Calculate Loss
3. Zero Gradients
4. Backpropagation
5. Update Weights

Pseudo-code:

prediction = model(X)

loss = loss_fn(prediction, y)

optimizer.zero_grad()

loss.backward()

optimizer.step()

---

## 16. Why do we use optimizer.zero_grad()?

**Answer:**

PyTorch accumulates gradients by default.

optimizer.zero_grad() clears old gradients before calculating new ones.

---

## 17. Why do we use loss.backward()?

**Answer:**

loss.backward() calculates gradients for all trainable parameters using backpropagation.

---

## 18. Why do we use optimizer.step()?

**Answer:**

optimizer.step() updates the model weights using the calculated gradients.

---

## 19. What is an Epoch?

**Answer:**

An Epoch is one complete pass through the entire training dataset.

Example:

1000 images

1 Epoch = Model sees all 1000 images once.

---

## 20. What is Batch Size?

**Answer:**

Batch Size is the number of samples processed before updating weights.

Example:

Dataset = 1000

Batch Size = 100

Weight updates = 10

---

## Interview Tip

Remember this sequence:

Input
↓
Forward Pass
↓
Prediction
↓
Loss
↓
Backward Pass
↓
Update Weights