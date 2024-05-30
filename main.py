import mlp_CIFAR10

def main():
    n_epochs = 4
    log_interval = 10

    cl = Classifier(n_epochs)
    cl.test()

if __name__ == '__main__':
    main()