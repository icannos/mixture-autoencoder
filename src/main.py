import argparse
import tensorflow as tf
from helpers import fetch_matlab_data, savemat

from mixtureautoencoder import mixture_autoencoder

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Mixture Autoencoder model")

    argparser.add_argument("--input-train", type=str,
                           help=".mat file to open. Should contain an X matrix")

    argparser.add_argument("--input-predict", type=str,
                           help=".mat file to open. Should contain an X matrix")

    argparser.add_argument("--output", type=str,
                           help="Where to store the results of the prediction of X_test, the file will a contain a"
                                "results array.")

    argparser.add_argument("--save-model-file", nargs="+", type=str,
                           help="File to dump weights after training, if training steps > 0")

    argparser.add_argument("--load-model-file", nargs="+", type=str,
                           help="File from which load weigths")

    argparser.add_argument("--training-steps", nargs="+", type=int, default=0,
                           help="Number of training steps to perform")

    argparser.add_argument("--autoencoder-topology", nargs="+", type=int,
                           help="Dimension of each hidden layer (only one side, the rest is built by symetry")
    argparser.add_argument("--classifier-topology", nargs="+", type=int,
                           help="Dimension of each hidden layer of the classifier network")
    argparser.add_argument("--input-dim", type=int, help="dimension of an entry vector")
    argparser.add_argument("--num-cluster", type=int, help="Number of expected clusters")
    argparser.add_argument("--autoencoders-activation", nargs="+", type=str,
                           help="Name of the activation function. Available: tanh sigmoid relu")

    argparser.add_argument("--entropy-strategy", type=str,
                           help="Strategy to use to define weights of sample entropy and batch entropy", default="balanced")

    args = argparser.parse_args()

    model = mixture_autoencoder(autoencoders_topology=tuple(args.autoencodertopology),
                                classifier_topology=tuple(args.classifiertopology),
                                autoencoders_activation=tuple(args.autoencodersactivation),
                                input_dim=args.inputdim,
                                num_clusters=args.numcluster)


    model.compile()

    model.init_session()


    ### Loading data
    data = fetch_matlab_data(args.input)

    if args.loadmodelfile is not None:
        model.saver.restore(model.sess, args.loadmodelfile)

    if args.inputtrain is not None:
        X_train = fetch_matlab_data(args.inputtrain)

        for _ in range(args.trainingstep):
            print(model.train(X_train, entropy_strategy="balanced"))

    if args.inputpredict is not None:
        X_test = fetch_matlab_data(args.inputpredict)

        if args.output is not None:
            savemat(args.output, {"results": model.predict(X_test)})
        else:
            print(model.predict(X_test))

    if args.savemodelfile is not None and args.trainingstep > 0:
        model.saver.save(model.save, args.savemodelfile)







