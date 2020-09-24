from random2 import seed, uniform, randint

class randomNum:

  def __init__(self, randSeed):
    seed(randSeed)

  def genRandom(self):
    return(uniform(0.1, 0.9) * (randint(0, 1) * 2 - 1))

def prependToArray(
  original,
  new
):
  original.insert(0, new)
  return(original)

def appendToArray(
  original,
  new
):
  original.insert(len(original), new)
  return(original)
