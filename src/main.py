import argparse
import tensorflow as tf

from mixtureautoencoder import mixture_autoencoder

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Mixture Autoencoder model")

    argparser.add_argument("--input", nargs="+", type=int,
                           help=".mat file to open, should contain an array of shape (num_data, data_dim) named X")

    argparser.add_argument("--model-file", nargs="+", type=int,
                           help="File to dump weights after training")

    argparser.add_argument("--autoencoder-topology", nargs="+", type=int,
                           help="Dimension of each hidden layer (only one side, the rest is built by symetry")
    argparser.add_argument("--classifier-topology", nargs="+", type=int,
                           help="Dimension of each hidden layer of the classifier network")
    argparser.add_argument("--input-dim", type=int, help="dimension of an entry vector")
    argparser.add_argument("--num-cluster", type=int, help="Number of expected clusters")
    argparser.add_argument("--autoencoders-activation", nargs="+", type=str,
                           help="Name of the activation function. Available: tanh sigmoid relu")

    argparser.add_argument("--entropy-strategy", type=str,
                           help="Strategy to use to define weights of sample entropy and batch entropy")

    args = argparser.parse_args()

    model = mixture_autoencoder(autoencoders_topology=tuple(args.autoencodertopology),
                                classifier_topology=tuple(args.classifiertopology),
                                autoencoders_activation=tuple(args.autoencodersactivation),
                                input_dim=args.inputdim,
                                num_clusters=args.numcluster)

    model.compile()
