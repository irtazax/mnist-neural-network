import neural_network as nn
import dataset as data


def main():
    nn.train(
        data.X_train,
        data.Y_train,
        data.X_test,
        data.Y_test,
        n_hidden_nodes=128,
        epochs=20,
        batch_size=128,
        lr=0.01
    )


if __name__ == "__main__":
    main()