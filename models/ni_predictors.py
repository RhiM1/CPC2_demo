import torch
import torch.nn.functional as F
from torch import Tensor, nn

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, input_dim, output_dim = 1):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, 2 * input_dim)
        self.linear2 = nn.Linear(2 * input_dim, 1)
        
        self.linear3 = nn.Linear(input_dim, output_dim)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: Tensor):
        
        # x has dim (*, time, feats)
        # att has dim (*, time, 1)
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))

        # att has new time (*, feats, time)
        att = att.transpose(-1,-2)
        att = F.softmax(att, dim = -1)

        # x has new dim (*, 1, feats)
        x = torch.bmm(att, x) 
        # x has new dim (*, feats)
        x = x.squeeze(1)
        
        # x has new dim (*, dim_out)
        x = self.linear3(x)
        
        return x  

