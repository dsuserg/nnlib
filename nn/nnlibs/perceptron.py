#!/usr/bin/env python3

import numpy as np
if __name__ == "__main__":
    import nnlib as nn
else:
    from . import nnlib as nn


def sigmoid(net, a=1.0):
    return 1/(1 + np.exp(net*-a))


def sigmoid_derivative(net, a=1.0):
    sigm = sigmoid(net)
    return a * sigm * (1 - sigm)


sgmoidFunc = nn.ActivationFunc(sigmoid, sigmoid_derivative)


class HiddenLayer(nn.HiddenLayer):

    def __init__(self, neuronsInLayer, connectedWith, dtype=np.float64):
        nn.HiddenLayer.__init__(self, neuronsInLayer, connectedWith, dtype)
        self.order = 1

    def init_сoefficients(self):
        self._neurons_coefficients = []
        coeff = np.random.rand(len(self._connected_with),
                               self._neurons_cnt
                               ).astype(self._dtype)
        self._neurons_coefficients.append(coeff)

        self._bias_coefficients = []
        biases = np.random.rand(self._neurons_cnt).astype(self._dtype)
        self._bias_coefficients.append(biases)

    def calc_neurons_state(self):
        order = self.order
        self._neurons *= 0
        for i in range(order):
            self._neurons += np.dot(
                self._connected_with.get_neurons()**(i+1),
                self._neurons_coefficients[i])

        self._neurons += self._bias_coefficients[0]  # with bias

    def set_new_order(self, order, rand=False):
        def generate_zeroes():
            return np.zeros([len(self._connected_with), len(self)],
                            dtype=self._dtype
                            )

        def generate_rnd():
            return np.random.rand(len(self._connected_with),
                                  self._neurons_cnt
                                  ).astype(self._dtype)

        if rand:
            generate = generate_rnd
        else:
            generate = generate_zeroes

        if order > self.order:
            #            init_w = self._neurons_coefficients[0]
            for i in range(order - self.order):
                init_w = generate()
                self._neurons_coefficients.append(init_w)
        else:
            self._neurons_coefficients = self._neurons_coefficients[:order - 1]

        self.order = order

    def get_biases(self):
        return self._bias_coefficients


class NPecrep(nn.NeuralNetwork):

    def __init__(self, structure, activation_func, dtype=np.float64):
        nn.NeuralNetwork.__init__(self, structure, activation_func, dtype)
        structure = structure[1:]
        self._order = 1
        fst_hidden = HiddenLayer(structure[0], self._input_layer, self._dtype)
        self._layers.append(fst_hidden)

        for i in range(1, len(structure)):
            hidden = HiddenLayer(structure[i], self._layers[i-1], self._dtype)
            self._layers.append(hidden)

        [layer.init_сoefficients() for layer in self._layers]

    def set_new_order(self, order, rand=False):
        self._order = order
        for layer in self._layers:
            layer.set_new_order(order, rand)

    def get_order(self):
        return self._order


class Backpropagation(nn.Trainer):

    def __init__(self, epoches_cnt, train_speed=1.0):
        self.epoches_cnt = epoches_cnt
        self.train_speed = train_speed

    def calc_discrepancies(self, discrepancies, errors, network):
        derivative = network.get_activation_func().derivative

        discrepancies[-1] = (derivative(network[-1].get_state_neurons())
                             * errors)

        for i in range(len(network) - 2, -1, -1):
            derivatives = derivative(network[i].get_state_neurons())
            weights = network[i+1].get_coefficients()[0]
            discrepancies[i] = (np.dot(discrepancies[i + 1],
                                       np.transpose(weights))
                                * derivatives)

    def update_coefficients(self, discrepancies, example, network):
        tr_speed = self.train_speed
        # for 1st layer special rules
        weights = network[0].get_coefficients()[0]
        weights += tr_speed*discrepancies[0]*(example[:, np.newaxis])

        for i in range(1, len(network)):
            weights = network[i].get_coefficients()[0]
            weights += (tr_speed
                        * discrepancies[i]
                        * (network[i-1].get_neurons()[:, np.newaxis])
                        )

        for i in range(len(network)):
            biases = network[i].get_biases()[0]
            biases += tr_speed*discrepancies[i]

    def train(self, neural_network, training_dataset):
        network = neural_network
        n_structure = network.get_structure()
        dtype = network.dtype()
        inp = training_dataset[0]
        req_out = training_dataset[1]
        nn_out = np.array(req_out)
        exmpls_cnt = len(inp)
        discrepancies = [np.zeros(i, dtype) for i in n_structure[1:]]

        for i in range(self.epoches_cnt):
            for j in range(exmpls_cnt):
                out = network.predict(inp[j])
                nn_out[j] = out
                errors = req_out[j] - nn_out[j]
                self.calc_discrepancies(discrepancies, errors, neural_network)
                self.update_coefficients(discrepancies,
                                         inp[j],
                                         neural_network)


class Backpropagation_nn(nn.Trainer):

    def __init__(self, epoches_cnt, train_speed=1.0):
        self.epoches_cnt = epoches_cnt
        self.train_speed = train_speed

    def calc_discrepancies(self, discrepancies, errors, network):
        derivative = network.get_activation_func().derivative

        discrepancies[-1] = (derivative(network[-1].get_state_neurons())
                             * errors)

        for i in range(len(network) - 2, -1, -1):
            derivatives = derivative(network[i].get_state_neurons())
            weights = network[i+1].get_coefficients()[0]
            discrepancies[i] = np.dot(discrepancies[i + 1],
                                      np.transpose(weights))
            for j in range(1, network.get_order()):
                weights = network[i+1].get_coefficients()[j]
                discrepancies[i] += np.dot(discrepancies[i + 1],
                                           np.transpose(weights))
            discrepancies[i] *= derivatives

    def update_coefficients(self, discrepancies, example, network):
        tr_speed = self.train_speed
        # for 1st layer special rules
        net_order = network.get_order()
        for j in range(net_order):
            weights = network[0].get_coefficients()[j]
            weights += tr_speed*discrepancies[0] * \
                (example[:, np.newaxis])**(j+1)

        for i in range(1, len(network)):
            for j in range(net_order):
                weights = network[i].get_coefficients()[j]
                weights += (tr_speed
                            * discrepancies[i]
                            * ((network[i-1].get_neurons()[:, np.newaxis])**(j+1))
                            )

        for i in range(len(network)):
            biases = network[i].get_biases()[0]
            biases += tr_speed*discrepancies[i]

    def train(self, neural_network, training_dataset):
        network = neural_network
        n_structure = network.get_structure()
        dtype = network.dtype()
        inp = training_dataset[0]
        req_out = training_dataset[1]
        nn_out = np.array(req_out)
        exmpls_cnt = len(inp)
        discrepancies = [np.zeros(i, dtype) for i in n_structure[1:]]

        for i in range(self.epoches_cnt):
            for j in range(exmpls_cnt):
                out = network.predict(inp[j])
                nn_out[j] = out
                errors = req_out[j] - nn_out[j]
                self.calc_discrepancies(discrepancies, errors, neural_network)
                self.update_coefficients(discrepancies,
                                         inp[j],
                                         neural_network)

