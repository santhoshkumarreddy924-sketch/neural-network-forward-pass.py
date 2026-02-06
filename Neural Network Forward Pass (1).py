#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

# Define Activation Functions [cite: 28]
def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# 1. Input Layer [cite: 30]
inputs = [1.0, 0.5]
print(f"Input Layer: {inputs}")

# 2. Hidden Layer Parameters (Hardcoded Weights and Biases) [cite: 27, 31]
# Neuron 1 weights: [0.5, 0.6], bias: 0.1
# Neuron 2 weights: [0.3, 0.1], bias: 0.1
weights_hidden = [[0.5, 0.6], [0.3, 0.1]]
biases_hidden = [0.1, 0.1]

# 3. Hidden Layer Computation [cite: 16, 17, 18]
hidden_layer_outputs = []

# Neuron 1
net1 = (inputs[0] * weights_hidden[0][0]) + (inputs[1] * weights_hidden[0][1]) + biases_hidden[0]
act1 = relu(net1)
print(f"Hidden Neuron 1 Net = {net1:.4f}, Activated (ReLU) = {act1:.4f}")

# Neuron 2
net2 = (inputs[0] * weights_hidden[1][0]) + (inputs[1] * weights_hidden[1][1]) + biases_hidden[1]
act2 = relu(net2)
print(f"Hidden Neuron 2 Net = {net2:.4f}, Activated (ReLU) = {act2:.4f}")

hidden_layer_outputs = [act1, act2]
print(f"Hidden Layer Outputs: {hidden_layer_outputs}")

# 4. Output Layer Computation [cite: 19, 32]
# Weights: [0.7, 0.8], bias: 0.2
weights_output = [0.7, 0.8]
bias_output = 0.2

net_out = (hidden_layer_outputs[0] * weights_output[0]) + (hidden_layer_outputs[1] * weights_output[1]) + bias_output
final_output = sigmoid(net_out)

print(f"\nOutput Neuron Net = {net_out:.4f}")
print(f"Final Output (Sigmoid) = {final_output:.4f}")


# In[ ]:




