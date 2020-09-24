from __init__ import Perceptron

import tests.main

def main():
  tests.main.test(Perceptron(
    mode = 'feedback',
    hidden_layers = (4,3),
    c = 0.01,
    epochs = 200,
    randSeed = 123
  ))

if __name__ == '__main__':
  main()
