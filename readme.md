

## Mixture Autoencoder model

This is an implementation of the model described in this paper Mixture Autoencoder from https://arxiv.org/abs/1712.07788 by D.Zhang.

## Usage

``
python3 main.py --input tests/
``

```
usage: Mixture Autoencoder model [-h] [--input-train INPUT_TRAIN]
                                 [--input-predict INPUT_PREDICT]
                                 [--output OUTPUT]
                                 [--save-model-file SAVE_MODEL_FILE]
                                 [--load-model-file LOAD_MODEL_FILE]
                                 [--training-steps TRAINING_STEPS]
                                 [--autoencoder-topology AUTOENCODER_TOPOLOGY [AUTOENCODER_TOPOLOGY ...]]
                                 [--classifier-topology CLASSIFIER_TOPOLOGY [CLASSIFIER_TOPOLOGY ...]]
                                 [--input-dim INPUT_DIM]
                                 [--num-clusters NUM_CLUSTERS]
                                 [--autoencoders-activation AUTOENCODERS_ACTIVATION [AUTOENCODERS_ACTIVATION ...]]
                                 [--entropy-strategy ENTROPY_STRATEGY]

optional arguments:
  -h, --help            show this help message and exit
  --input-train INPUT_TRAIN
                        .mat file to open. Should contain an X matrix
  --input-predict INPUT_PREDICT
                        .mat file to open. Should contain an X matrix
  --output OUTPUT       Where to store the results of the prediction of
                        X_test, the file will a contain aresults array.
  --save-model-file SAVE_MODEL_FILE
                        File to dump weights after training, if training steps
                        > 0
  --load-model-file LOAD_MODEL_FILE
                        File from which load weigths
  --training-steps TRAINING_STEPS
                        Number of training steps to perform
  --autoencoder-topology AUTOENCODER_TOPOLOGY [AUTOENCODER_TOPOLOGY ...]
                        Dimension of each hidden layer (only one side, the
                        rest is built by symetry
  --classifier-topology CLASSIFIER_TOPOLOGY [CLASSIFIER_TOPOLOGY ...]
                        Dimension of each hidden layer of the classifier
                        network
  --input-dim INPUT_DIM
                        dimension of an entry vector
  --num-clusters NUM_CLUSTERS
                        Number of expected clusters
  --autoencoders-activation AUTOENCODERS_ACTIVATION [AUTOENCODERS_ACTIVATION ...]
                        Name of the activation function. Available: tanh
                        sigmoid relu
  --entropy-strategy ENTROPY_STRATEGY
                        Strategy to use to define weights of sample entropy
                        and batch entropy

```
