import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mlp_cifar import Classifier

def main():
    n_epochs = 4
    log_interval = 10

    cl = Classifier(n_epochs)
    cl.test()

if __name__ == '__main__':
    main()