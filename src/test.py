import pp

from Perceptron import Perceptron

class TestPerceptron:

  def __init__(self):
    print('\n\nInit:')
    self.predictor = Perceptron(
      mode = 'feedback',
      hidden_layers = (2, 2),
      c = 0.1,
      epochs = 200,
      seed = 123
    )
    pp(vars(self.predictor))

  def test_train(self):
    print('\n\nTraining:')
    self.predictor.train([
      [0, 0],
      [1, 1],
      [0, 1],
      [1, 0]
    ], ['hi', 'bye', 'hi', 'bye'])
    pp(vars(self.predictor))

  def test_predict(self):
    print('\n\nPrediction:')
    pp(self.predictor.predict([
      [0, 0],
      [1, 1],
      [0, 1],
      [1, 0]
    ]))

  def runall(self):
    self.test_train()
    self.test_predict()

if __name__ == '__main__':
  test = TestPerceptron()
  test.runall()
