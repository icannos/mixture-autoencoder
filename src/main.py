import argparse
import tensorflow as tf
from helpers import fetch_matlab_data, savemat, fetch_matlab_data_clusters

from mixtureautoencoder import mixture_autoencoder

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Mixture Autoencoder model")

    ### io parameters

    argparser.add_argument("--input-train", type=str,
                           help=".mat file to open. Should contain an X matrix, if pretrain then the file should contain X_0 .. X_[num_cluster-1].")

    argparser.add_argument("--input-predict", type=str,
                           help=".mat file to open. Should contain an X matrix")

    argparser.add_argument("--output", type=str,
                           help="Where to store the results of the prediction of X_test, the file will a contain a"
                                "results array.")

    argparser.add_argument("--save-model-file", type=str,
                           help="File to dump weights after training, if training steps > 0")

    argparser.add_argument("--load-model-file",  type=str,
                           help="File from which load weigths")

    ### Topology parameters

    argparser.add_argument("--num-clusters", type=int, help="Number of expected clusters")

    argparser.add_argument("--input-dim", type=int, help="dimension of an entry vector")

    argparser.add_argument("--autoencoder-topology", nargs="+", type=int,
                           help="Dimension of each hidden layer (only one side, the rest is built by symetry")

    argparser.add_argument("--classifier-topology", nargs="+", type=int,
                           help="Dimension of each hidden layer of the classifier network")

    argparser.add_argument("--autoencoders-activation", nargs="+", type=str, choices=["tanh", "sigmoid"],
                           help="Name of the activation function. Available: tanh sigmoid relu")

    ### Training hyper

    argparser.add_argument("--pre-train-steps", type=int, default=0,
                           help="Will pretrain separatly each subautoencoded. Useful when it is difficult to assign data to different cluster")

    argparser.add_argument("--training-steps", type=int, default=0,
                           help="Number of training steps to perform")

    argparser.add_argument("--entropy-strategy", type=str,
                           help="Strategy to use to define weights of sample entropy and batch entropy",
                           default="balanced")
    argparser.add_argument("--batch-size", type=int, default=64,
                           help="Size of a training batch")
    argparser.add_argument("--hp-bentropy", type=float, default=1,
                           help="Coefficient to apply to the sample entropy term")
    argparser.add_argument("--hp-sentropy", type=float, default=1,
                           help="Coefficient to apply to the batch entropy term")
    argparser.add_argument("--hp-encoded-reg", type=float, default=0.005,
                           help="Weigth of the regularization term")

    args = argparser.parse_args()


    activations = list()

    for s in args.autoencoders_activation:
        if s == "tanh":
            activations.append(tf.nn.tanh)
        if s == "sigmoid":
            activations.append(tf.nn.sigmoid)

    model = mixture_autoencoder(batch_size=args.batch_size,
                                autoencoders_topology=tuple(args.autoencoder_topology),
                                classifier_topology=tuple(args.classifier_topology),
                                autoencoders_activation=activations,
                                input_dim=args.input_dim,
                                num_clusters=args.num_clusters,
                                hyper_param_bentropy=args.hp_bentropy,
                                hyper_param_sentropy=args.hp_sentropy,
                                hp_encoded_reg=args.hp_encoded_reg)



    model.compile()

    model.init_session()



    ### Loading data

    if args.load_model_file is not None:
        model.saver.restore(model.sess, args.load_model_file)

    if args.input_train is not None and args.pre_train_steps > 0:
        X_train = fetch_matlab_data_clusters(args.input_train, args.num_clusters)

        for k in range(args.num_clusters):
            print(model.pretrain(X_train[k], k))

    if args.input_train is not None and args.training_steps > 0:
        X_train = fetch_matlab_data(args.input_train)

        for _ in range(args.training_steps):
            loss, batch_wise_entropy, p_mean  = model.train(X_train, entropy_strategy="balanced")

            print("Global_Loss {loss}, Batch entropy: {batch_wise_entropy}, Cluster mean probability: {p_mean}".format(
                loss=loss, batch_wise_entropy=batch_wise_entropy, p_mean=p_mean))


    if args.input_predict is not None:
        X_test = fetch_matlab_data(args.input_predict)

        if args.output is not None:
            savemat(args.output, {"results": model.predict(X_test)})
        else:
            print(model.predict(X_test))

    if args.save_model_file is not None and args.training_steps > 0:
        model.saver.save(model.sess, args.save_model_file)







