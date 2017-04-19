# tf-regression
## Overview
A simple implementation for regression problems using Python 2.7 and TensorFlow

* [train_test_data.py](train_test_data.py) is a script that preprocesses a data file into the necessary train and test set arrays for TensorFlow. It includes functions to convert categorical variables into dummies, convert string values into Python compatible strings, and remove outliers

* [train_neural_network.py](train_neural_network.py) contains the steps to build and evaluate a TensorFlow neural network. Bulk of code from PythonProgramming.net with further enhancements including cost function tracking, leaky ReLU implementation, and elastic net regularisation (from [TensorFlow Machine Learning Cookbook](https://github.com/nfmcclure/tensorflow_cookbook) by @nfmcclure)

## Resources
Here are some additional resources if you are looking to explore neural networks and TensorFlow more extensively:

### Tensorflow and Neural Networks
1. [Deep Learning with Neural Networks and TensorFlow series by PythonProgramming.net](https://pythonprogramming.net/neural-networks-machine-learning-tutorial/)
2. [MNIST for ML Beginners by TensorFlow.org](https://www.tensorflow.org/get_started/mnist/beginners)
3. [Calculus on Computational Graphs: Backpropagation by @colah](http://colah.github.io/posts/2015-08-Backprop/) 
4. [TensorFlow Tutorials by @pkmital](https://github.com/pkmital/tensorflow_tutorials) 
5. [Neural Networks, Manifolds, and Topology by @colah](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
6. [Implementing a Neural Network from Scratch by @dennybritz](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)
7. [Efficient Backprop by LeCun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) 
8. [Practical Recommendations for Gradient-Based Training of Deep Architectures by Bengio, 2012](https://arxiv.org/abs/1206.5533)

### TensorBoard
1. [Simple Introduction to Tensorboard Embedding Visualisation](http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/)
