import numpy as np

"""
  validateInputs takes two dictionaries as inputs:
    1) a dictionary of function inputs
    2) a dictionary of types
  both dictionaries must have matching keys and keys must be unique; validateInputs checks for that

  returns None and throws an error if input dictionary types don't match the type dictionary's types

  if the type is a list, it is treated as an enum:
    - the enum types are validated against one another
    - enum is checked for presence of value
"""

def validateInputs(
  inputs,
  types
):
  inputKeys = inputs.keys()
  typeKeys = types.keys()

  sameLength = len(inputKeys) == len(typeKeys)
  uniqueInputs = len(set(inputKeys)) == len(inputKeys)
  uniqueTypes = len(set(typeKeys)) == len(typeKeys)
  stringInputs = all([ type(key) == str for key in inputKeys ])
  stringTypes = all([ type(key) == str for key in typeKeys ])

  if sameLength and uniqueInputs and uniqueTypes and stringInputs and stringTypes:
    for key in inputKeys:
      if type(types[key]) == list:
        if inputs[key] not in types[key]:
          raise TypeError('Input types are invalid.')
      else:
        if type(inputs[key]) != types[key]:
          raise TypeError('Input types are invalid.')
  else:
    raise ValueError('Type checking inputs have invalid keys.')

"""
  validateForTraining takes two ndarrays as inputs:
    1) an ndarray of the feature space
    2) an ndarray of the target variable
  both ndarrays must have the same non-zero length and must all have int or float types;
    validateForTraining for that

  returns None and throws an error if input ndarrays are invalid
"""

def validateForTraining(
  x,
  y
):
  if len(x) != len(y):
    raise ValueError('Feature space and target variable must have an equal number of records.')

  if len(x) == 0:
    raise ValueError('You did not provide any records.')

  if len(x[0]) == 0:
    raise ValueError('You did not provide any features.')

  if not all([ all([ type(value) == int or type(value) == float for value in row ]) for row in x ]):
    raise ValueError('Feature space must be int or float.')

  if not all([ type(value) == str for value in y ]):
    raise ValueError('Target variable must be string.')
