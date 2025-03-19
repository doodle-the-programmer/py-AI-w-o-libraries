import tkinter as tk
import random
import math

def softmax(x):
    e_x = [math.exp(i) for i in x]
    sum_e_x = sum(e_x)
    return [i / sum_e_x for i in e_x]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.output = None
        self.inputs = None

    def forward(self, inputs, activation_function):
        self.inputs = inputs
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = activation_function(weighted_sum)
        return self.output

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # Derivative of sigmoid function

    def relu_derivative(self, x):
        return 1 if x > 0 else 0  # Derivative of ReLU

class Layer:
    def __init__(self, num_input_neurons, num_output_neurons, activation_function):
        self.neurons = [Neuron(num_input_neurons) for _ in range(num_output_neurons)]
        self.activation_function = activation_function

    def forward(self, inputs):
        raw_outputs = [neuron.forward(inputs, self.activation_function) for neuron in self.neurons]
        if self.activation_function == sigmoid:
            return softmax(raw_outputs)
        return raw_outputs

    def get_outputs(self):
        return [neuron.output for neuron in self.neurons]

class NeuralNetwork:
    def __init__(self, layer_sizes, mutation_strength=0.1):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i + 1], relu if i < len(layer_sizes) - 2 else sigmoid)
                       for i in range(len(layer_sizes) - 1)]
        self.mutation_strength = mutation_strength

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, inputs, target, learning_rate=0.3):
        output = self.forward(inputs)
        errors = [target[i] - output[i] for i in range(len(target))]

        for layer_index in reversed(range(len(self.layers))):
            layer = self.layers[layer_index]
            if layer_index == len(self.layers) - 1:
                for i, neuron in enumerate(layer.neurons):
                    output_derivative = neuron.sigmoid_derivative(neuron.output)
                    neuron.weights = [w + learning_rate * errors[i] * output_derivative * input_val
                                      for w, input_val in zip(neuron.weights, neuron.inputs)]
                    neuron.bias += learning_rate * errors[i] * output_derivative
            else:
                for i, neuron in enumerate(layer.neurons):
                    pass

# Experimental code
layer_sizes = [9, 8, 3]

# Example training data
training_data = [
    [[1, 0, 0, 1, 0, 0, 1, 1, 1], [1, 0, 0]],  # 0 corresponds to "L" shapes
    [[1, 0, 0, 1, 0, 0, 1, 1, 0], [1, 0, 0]],
    [[0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 0]],
    [[1, 1, 1, 1, 0, 1, 1, 1, 1], [0, 1, 0]],  # 1 corresponds to "O" shapes
    [[1, 1, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0]],
    [[0, 1, 1, 0, 1, 1, 0, 0, 0], [0, 1, 0]],
    [[0, 0, 0, 1, 1, 0, 1, 1, 0], [0, 1, 0]],
    [[0, 0, 0, 0, 1, 1, 0, 1, 1], [0, 1, 0]],
    [[1, 1, 0, 0, 1, 1, 1, 1, 0], [0, 0, 1]],  # 2 corresponds to ">" shapes
    [[1, 0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1]],
    [[0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 0, 1]],
    [[1, 1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 1]],
    [[1, 0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1]],
]

# Training the network
class NeuralNetworkEnsemble:
    def __init__(self, num_networks, layer_sizes):
        self.networks = [NeuralNetwork(layer_sizes) for _ in range(num_networks)]

    def forward(self, inputs):
        outputs = [network.forward(inputs) for network in self.networks]
        averaged_output = [sum(output[i] for output in outputs) / len(outputs) for i in range(len(outputs[0]))]
        return averaged_output


# Create an ensemble of 5 networks
ensemble = NeuralNetworkEnsemble(10, layer_sizes)

# Training all networks in the ensemble
for epoch in range(1000):
    best_error = float("inf")
    for data in training_data:
        inputs, target = data
        for network in ensemble.networks:
            output = network.forward(inputs)
            network.backward(inputs, target)
            error = sum([(output[i] - target[i]) ** 2 for i in range(len(output))])
            if error < best_error:
                best_error = error
    print(f"Epoch {epoch}, Best Error: {best_error}")

# Tkinter setup
class ShapeDrawer(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Shape Drawer")
        self.canvas = tk.Canvas(self, width=300, height=300)
        self.canvas.pack()

        self.cells = []
        cell_size = 100
        for row in range(3):
            row_cells = []
            for col in range(3):
                cell = self.canvas.create_rectangle(col * cell_size, row * cell_size,
                                                     (col + 1) * cell_size, (row + 1) * cell_size,
                                                     fill="white", outline="black")
                row_cells.append(cell)
            self.cells.append(row_cells)

        self.current_state = [[0 for _ in range(3)] for _ in range(3)]

        self.canvas.bind("<Button-1>", self.toggle_cell)

        self.predict_button = tk.Button(self, text="Predict", command=self.predict_shape)
        self.predict_button.pack()

    def toggle_cell(self, event):
        cell_size = 100
        row = event.y // cell_size
        col = event.x // cell_size
        if self.current_state[row][col] == 0:
            self.canvas.itemconfig(self.cells[row][col], fill="black")
            self.current_state[row][col] = 1
        else:
            self.canvas.itemconfig(self.cells[row][col], fill="white")
            self.current_state[row][col] = 0

    def predict_shape(self):
        input_data = [self.current_state[row][col] for row in range(3) for col in range(3)]
        output = ensemble.forward(input_data)

        predicted_shape = output.index(max(output))
        shape_names = ["L", "O", ">"]
        certainty = max(output)

        print(f"Predicted shape: {shape_names[predicted_shape]}")
        print(f"Certainty: {certainty:.2f}")


app = ShapeDrawer()
app.mainloop()