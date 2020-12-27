# Deep Learning and Neural Networks Example

Examples of NN models. DL algorithms

## Task description
Given test and train data contains images of cats and non-cats. Need to create the model, which defines a cat picture(1) and a non-cat picture(0).

### Outputs:
* For two-layer network:
  
  **(there is a bird picture)**
  
  ```
  y = 0. It's a non-cat picture.

  Number of training examples: 209
  Number of testing examples: 50
  
  Each image is of size: (64, 64, 3)
  train_x_orig shape: (209, 64, 64, 3)
  train_y shape: (1, 209)
  test_x_orig shape: (50, 64, 64, 3)
  test_y shape: (1, 50)
  train_x's shape: (12288, 209)
  test_x's shape: (12288, 50)

  Cost after iteration 0: 0.693049735659989
  Cost after iteration 100: 0.6464320953428849
  ...                       ...
  Cost after iteration 2400: 0.04855478562877019

  Accuracy: 0.9999999999999998
  Accuracy: 0.72```
  
* For L-layer network (4-layer):
  
  **(there is a bird picture)**
  
  ```
  y = 0. It's a non-cat picture.
  
  Number of training examples: 209
  Number of testing examples: 50
  
  Each image is of size: (64, 64, 3)
  train_x_orig shape: (209, 64, 64, 3)
  train_y shape: (1, 209)
  test_x_orig shape: (50, 64, 64, 3)
  test_y shape: (1, 50)
  train_x's shape: (12288, 209)
  test_x's shape: (12288, 50)
  
  Cost after iteration 0: 0.771749
  Cost after iteration 100: 0.672053
  ...                       ...
  Cost after iteration 2400: 0.092878
  
  Accuracy: 0.985645933014
  Accuracy: 0.8```
