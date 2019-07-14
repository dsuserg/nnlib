#!/usr/bin/env python3

import numpy as np


class Dataset:

    def __init__(self, data, dtype=np.float64):
        self._data = np.array(data, dtype=dtype)
        self._size = len(self._data)
        self._example_size = len(self._data[0])
        self._status = "pure"
        self._range = None
        self._expert = None
        self._percentage = None
        self._max_elems = None
        self._min_elems = None
        self._centers = None

    def tune_up(self, expert=None, percentage=None):
        self._max_elems = np.zeros((self._example_size))
        self._min_elems = np.zeros((self._example_size))
        self._centers = np.zeros((self._example_size))

        if expert:
            for i in range(self._example_size):
                if expert[i]:
                    self._min_elems[i] = expert[i][0]
                    self._min_elems[i] = expert[i][1]
                else:
                    self._max_elems[i] = max(self._data[:, i])
                    self._min_elems[i] = min(self._data[:, i])

        else:
            for i in range(self._example_size):
                self._max_elems[i] = max(self._data[:, i])
                self._min_elems[i] = min(self._data[:, i])

        if percentage:
            for i in range(self._example_size):
                if percentage[i]:
                    self._max_elems[i] += percentage[i]*(self._max_elems[i]
                                                         - self._min_elems[i])

                    self._min_elems[i] -= percentage[i]*(self._max_elems[i]
                                                         - self._min_elems[i])
                else:
                    continue

        for i in range(self._example_size):
            self._centers[i] = (self._max_elems[i] + self._min_elems[i])/2

    def normalisation_linear(self, rng):
        if rng[0]:
            for i in range(self._size):
                for j in range(self._example_size):
                    numerator = 2*(self._data[i][j] - self._min_elems[j])
                    denominator = self._max_elems[j] - self._min_elems[j]
                    self._data[i][j] = numerator/denominator - 1

        else:
            for i in range(self._size):
                for j in range(self._example_size):
                    numerator = self._data[i][j] - self._min_elems[j]
                    denominator = self._max_elems[j] - self._min_elems[j]
                    self._data[i][j] = numerator/denominator

    def normalisation_nonlinear(self, rng, alfa):
        if rng[0]:
            for i in range(self._size):
                for j in range(self._example_size):
                    degree = -alfa[j]*(self.data[i][j] - self._centers[j])
                    self._data[i][j] = (np.exp(degree)-1)/(np.exp(degree)+1)

        else:
            for i in range(self._size):
                for j in range(self._example_size):
                    degree = -alfa[j]*(self.data[i][j] - self._centers[j])
                    self._data[i][j] = 1/np.exp(degree) + 1

    def denormalisation_linear(self, rng, example):
        out = np.zeroes((self._example_size))
        if rng[0]:
            for i in range(self._example_size):
                substraction = self._max_elems[i] - self._min_elems[i]
                out[i] = self._min_elems[i] + (example[i] + 1)*substraction/2
        else:
            for i in range(self._example_size):
                substraction = self._max_elems[i] - self._min_elems[i]
                out[i] = self._min_elems[i] + example[i]*substraction

    def denormalisation_nonlinear(self, rng, alfa, example):
        out = np.zeroes((self._example_size))
        if rng[0]:
            for i in range(self._example_size):
                out[i] = self._centers[i] - \
                    (1/alfa[i]) * np.log(1/example[i]-1)
        else:
            for i in range(self._example_size):
                temp = (1-self._centers[i])/(1+self._centers[i])
                out[i] = self._centers[i] - (1/alfa[i]) * np.log(temp)

    def get_status(self):
        return self._status

    def example_size(self):
        return self._example_size

    def _prepare_data(self):
        self._mins = np.amin(self._data, axis=0)
        self._maxs = np.amax(self._data, axis=0)
        self._centers = [(self._maxs[i]+self._mins[i])/2
                         for i in range(self._example_size)]

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return self._size

    def __iter__(self):
        return (example for example in self._data)
