import torch
import torch.nn.functional as F
from torch import nn
from models.ni_predictors import PoolAttFF

 
class MetricPredictorLSTM_layers(nn.Module):
    """Metric estimator for enhancement training.

    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers

    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    """

    def __init__(
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU, att_pool_dim=512, num_layers = 12
    ):
        super().__init__()

        self.layer_weights = nn.Parameter(torch.ones(num_layers, dtype = torch.float))
        self.sm = nn.Softmax(dim = 0)

        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        self.attenPool = PoolAttFF(att_pool_dim)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):

        X = X @ self.sm(self.layer_weights)
        X, _ = self.blstm(X)
        X = self.attenPool(X)
        X = self.sigmoid(X)

        return X, None

