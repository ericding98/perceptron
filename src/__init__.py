import numpy as np

from random2 import seed, uniform

"""
  Title: Perceptron Predictor Library
  Description: A perceptron predictor library created for CISC 452 class at Queen's University in Kingston, ON, Canada
  Author: Eric Ding (eric.ding.98@gmail.com) | 20011628 | 15ed21
"""

"""
  The Perceptron class templates an instance of a Perceptron predictor.
"""
class Perceptron:

  """
    Perceptron is initialized with 5 parameters:
      - mode: 'feedback' or 'pocket'
      - hidden_layers: a tuple whose length encodes the # of hidden layers and whose values encodes the number of nodes per layer
      - c: the learning rate
      - epochs: the max number of epochs the model will train for
      - randSeed: a seed for random numbers
  """

  def __init__(
    self,
    mode = 'feedback',
    hidden_layers = (),
    c = 0.01,
    epochs = 200,
    randSeed = None
  ):
    # initialize seed
    self.seed = randSeed

    # initialize mode
    self.mode = mode

    # initialize hidden layers
    self.hidden_layers = hidden_layers
    self.n_hidden_layers = 0
    self.countHiddenLayers()

    # iniialize weights
    self.weights = np.array([])
    self.initializeWeights()

    # initialize training parameters
    self.c = c
    self.epochs = epochs

  """
    countHiddenLayers initializes the # of hidden layers to the length of the input tuple for hidden layers
  """

  def countHiddenLayers(self):
    self.n_hidden_layers = len(self.hidden_layers)

  """
    initializeWeights initializes weights to an ndarray representing the inner-weights
    values are randomized on [0.1, 0.9)
    initialized to empty array if there are no hidden layers or if there is one hidden layer
    for convenient, weights are defined in terms of the layer which is closer to the output layer
  """

  def initializeWeights(self):
    if self.n_hidden_layers > 1:

      #seeded
      seed(self.seed)

      n_inner_weight_layers = self.n_hidden_layers - 1

      self.weights = np.array([
        [
          [ uniform(0.1, 0.9) for _ in range(self.hidden_layers[i]) ]
          for _ in range(self.hidden_layers[i+1])
        ]
        for i in range(n_inner_weight_layers)
      ])

  """
    train takes x and y as ndarrays and updates Perceptron's weights according to the mode and epochs specified
  """

  def train(
    self,
    x,
    y
  ):
    
