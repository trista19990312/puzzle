import os
from NetworkHnn10 import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_file = "hnn_data.pkl.gz"

if __name__ == '__main__':
    # construct networks
    net = Network([
                    # Layer0:
                    FullyConnectedLayer(n_in=16, n_out=32),

                    # Layer1:
                    FullyConnectedLayer(n_in=32, n_out=32),

                    # Layer3:
                    LinearLayer(n_in=32, n_out=1)
                   ])       

    # train networks
    net.train(data_set_name=train_file)
