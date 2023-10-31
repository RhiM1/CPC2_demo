import torch
from torch import nn
from models.huBERT_wrapper import WhisperWrapper_full
# except:
#     from huBERT_wrapper import HuBERTWrapper_full,HuBERTWrapper_extractor
#     from wav2vec2_wrapper import Wav2Vec2Wrapper_no_helper,Wav2Vec2Wrapper_encoder_only
#     from llama_wrapper import LlamaWrapper
from speechbrain.processing.features import spectral_magnitude,STFT
# from models.ni_predictors import PoolAttFF
   
    
class WhisperFull_feats(nn.Module):

    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None):
        super().__init__()
        
        self.feat_extract = WhisperWrapper_full(
            layer = layer,
            pretrained_model = pretrained_model,
            use_feat_extractor = use_feat_extractor
        )

    def forward(self, x):

        x = self.feat_extract(x)#.permute(0,2,1)
        # print(f"whisperencoder_feats: {x.size()}")

        return x
    
