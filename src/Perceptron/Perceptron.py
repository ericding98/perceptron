from helpers import validateInputs, validateForTraining, randomNum, prependToArray, appendToArray

"""
  Title: Perceptron Predictor Library
  Description: A perceptron predictor library created for CISC 452 class at Queen's University in
    Kingston, ON, Canada
  Author: Eric Ding (eric.ding.98@gmail.com) | 20011628 | 15ed21

  See README for more information
"""

"""
  The Perceptron class templates an instance of a Perceptron predictor.
"""
class Perceptron:

  """
    Perceptron is initialized with 5 parameters:
      - mode: 'feedback' or 'pocket'
      - hidden_layers: a tuple whose length encodes the # of hidden layers and whose values
          encodes the number of nodes per layer
      - c: the learning rate
      - epochs: the max number of epochs the model will train for
      - seed: a seed for random numbers
  """

  def __init__(
    self,
    mode = 'feedback',
    hidden_layers = (),
    c = 0.1,
    epochs = 200,
    seed = None
  ):
    validateInputs({
      'mode': mode,
      'hidden_layers': hidden_layers,
      'c': c,
      'epochs': epochs,
      'seed': seed
    }, {
      'mode': [
        'feedback',
        'pocket'
      ],
      'hidden_layers': tuple,
      'c': float,
      'epochs': int,
      'seed': NoneType if seed == None else int
    })

    # initialize seed
    self.random = randomNum(seed)

    # initialize mode
    self.mode = mode

    # initialize hidden layers
    self.hidden_layers = hidden_layers
    self.n_hidden_layers = 0
    self.countHiddenLayers()

    # iniialize weights
    self.weights = [] # biases included
    self.initializeWeights()

    # initialize training parameters
    self.c = c
    self.epochs = epochs

    #initialize defaults
    self.encoding = {} # initialized and updated in train
    self.classes = [] # initialized and updated in train
    self.output_nodes = 0 # initialized in train; if 0, predict will throw error
    self.input_nodes = 0 # initialized in train; if later input layer dimensions for train or predict
                         # doesn't match then will throw error

  """
    countHiddenLayers initializes the # of hidden layers to the length of the input tuple for
      hidden layers
  """

  def countHiddenLayers(self):
    self.n_hidden_layers = len(self.hidden_layers)

  """
    initializeWeights initializes weights to an list representing the inner-weights
    values are randomized on [0.1, 0.9)
    initialized to empty array if there are no hidden layers or if there is one hidden layer
    for convenient, weights are defined in terms of the layer which is closer to the output layer
  """

  def initializeWeights(self):
    if self.n_hidden_layers > 1:

      n_inner_weight_layers = self.n_hidden_layers - 1

      self.weights = [
        [
          [ self.random.genRandom() for _ in range(self.hidden_layers[i] + 1) ]
          for _ in range(self.hidden_layers[i + 1])
        ]
        for i in range(n_inner_weight_layers)
      ]

  """
    train takes x and y as lists and updates Perceptron's weights according to the mode and
      epochs specified
  """

  def train(
    self,
    x,
    y
  ):
    validateInputs({
      'x': x,
      'y': y
    }, {
      'x': list,
      'y': list
    })

    validateForTraining(x, y)
    self.addNewClasses(y)
    self.updateOuterWeights(x, y)

    embeddings = self.encodeClasses(y)
    # loop through epochs
    for _ in range(self.epochs):
      # loop through records
      for i in range(len(x)):
        data = x[i]
        dataRecord = [] # to record input values for updating weights
        # loop through layers
        for weightLayer in self.weights:
          dataRecord.append(data)
          # sumproduct
          data = [
            sum([
              (1 if j == 0 else data[k-1]) * weightLayer[j][k]
              for k in range(len(weightLayer[j]))
            ])
            for j in range(len(weightLayer))
          ]
          # activation
          data = [
            1 if point >= 0 else 0
            for point in data
          ]
        # adjust weights for every output node
        self.weights = [
          [
            [
              self.weights[j][m][n] + self.c * (1 if n == 0 else dataRecord[j][n - 1]) * (1 if data[m] > embeddings[i][m] else -1 if data[m] < embeddings[i][m] else 0)
              for n in range(len(self.weights[j][m]))
            ]
            for m in range(len(self.weights[j]))
          ]
          for j in range(len(self.weights))
        ]

  """
    encodeClasses takes the target variable as input and returns a one-hot encoded
      representation
  """

  def encodeClasses(
    self,
    y
  ):
    self.encoding = {
      targetClass: [
        1 if i == self.classes.index(targetClass) else 0
        for i in range(len(self.classes))
      ]
      for targetClass in self.classes
    }
    return([
      self.encoding[trainingClass] for trainingClass in y
    ])

  """
    updateOuterWeights takes the feature space and target variable as inputs and
      1) if model is uninitialized, initializes weights for output and input layers
      2) if model is initialized, identify new classes and initialize their weights
  """

  def updateOuterWeights(
    self,
    x,
    y
  ):
    # if uninitialized
    if self.input_nodes == 0:
      # can assume x[0] exists and is non-zero in length
      self.initializeOuterWeights(len(x[0]), len(self.classes))
      self.input_nodes = len(x[0])
      self.output_nodes = len(self.classes)
    else:
      if self.input_nodes != len(x[0]):
        raise ValueError('You cannot introduce a new feature space.')
      else:
        newOutputNodes = len(self.classes) - self.output_nodes
        if newOuputNodes != 0:
          addOutputNodes(newOutputNodes)

  """
    addOutputNodes takes the number of new output nodes as an input and initializes weights for
      that number of nodes
  """

  def addOutputNodes(
    self,
    n
  ):
    numberOfNodesNearOutput = self.weights[-2]

    self.weights[-1].extend([
      [ self.random.genRandom() for _ in range(numberOfNodesNearOutput)]
      for _ in range(n)
    ])

  """
    initializeOuterWeights takes the number of input nodes and number of output nodes as inputs and
      initializes their weights
  """

  def initializeOuterWeights(
    self,
    n_input,
    n_output
  ):
    if len(self.hidden_layers) == 0:
      self.weights = appendToArray(
        self.weights,
        [
          [ self.random.genRandom() for _ in range(n_input + 1) ]
          for _ in range(n_output)
        ]
      )
    elif len(self.hidden_layers) == 1:
      self.weights = prependToArray(
        self.weights,
        [
          [ self.random.genRandom() for _ in range(n_input + 1) ]
          for _ in range(self.hidden_layers[0])
        ]
      )
      self.weights = appendToArray(
        self.weights,
        [
          [ self.random.genRandom() for _ in range(self.hidden_layers[0] + 1) ]
          for _ in range(n_output)
        ]
      )
    else:
      numberOfNodesNearInput = len(self.weights[0][0])
      numberOfNodesNearOutput = len(self.weights[-1])

      self.weights = prependToArray(
        self.weights,
        [
          [ self.random.genRandom() for _ in range(n_input + 1) ]
          for _ in range(numberOfNodesNearInput)
        ]
      )

      self.weights = appendToArray(
        self.weights,
        [
          [ self.random.genRandom() for _ in range(numberOfNodesNearOutput + 1) ]
          for _ in range(n_output)
        ]
      )

  """
    addNewClasses takes the target variable as an input and updates the model's in-memory classes
      with observed classes in the new training data that are new and unique.
  """

  def addNewClasses(
    self,
    y
  ):
    for targetClass in set(y):
      if targetClass not in self.classes:
        self.classes.append(targetClass)

  """
    predict takes a feature space as input and returns the predction using the weights
  """

  def predict(
    self,
    x
  ):
    predictions = []

    # loop through records
    for i in range(len(x)):
      data = x[i]
      # loop through layers
      for weightLayer in self.weights:
        # sumproduct
        data = [
          sum([
            (1 if j == 0 else data[k-1]) * weightLayer[j][k]
            for k in range(len(weightLayer[j]))
          ])
          for j in range(len(weightLayer))
        ]
        # activation
        data = [
          1 if point >= 0 else 0
          for point in data
        ]
      predictions.append(data)

    return(predictions)
