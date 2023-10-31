import argparse
import numpy as np
import torch
import torch.nn as nn
import speechbrain as sb
import torchaudio.transforms as T
from models.ni_predictor_models import MetricPredictorLSTM_layers
from models.ni_feat_extractors import WhisperFull_feats


def compute_feats(wavs,resampler):
    """Feature computation pipeline"""
    wavs_l = wavs[:, 0]
    wavs_r = wavs[:, 1]
    wavs_l = resampler(wavs_l)
    wavs_r = resampler(wavs_r)
    return wavs_l,wavs_r


def audio_pipeline(path,fs=32000):
    wavs = sb.dataio.dataio.read_audio_multichannel(path)    
    return wavs


def main(args):

    args.exemplar = False
 
    feat_extractor = WhisperFull_feats(
        pretrained_model=None, 
        use_feat_extractor = True,
        layer = -1
    )
    dim_extractor = 768
    hidden_size = 768//2
    activation = nn.LeakyReLU
    att_pool_dim = 768

    model = MetricPredictorLSTM_layers(dim_extractor, hidden_size, activation, att_pool_dim, num_layers = 12)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters]) 
    print(f"Number of parameters: {num_params}")

    # model.load_state_dict(torch.load(args.model_file))

    resampler = T.Resample(32000, 16000)
    wavs = audio_pipeline(args.input_audio)

    print(f"wavs size: {wavs.size()}")
    feats_l,feats_r = compute_feats(wavs,resampler)
    print(f"feats_l 1 size: {feats_l.size()}")
    
    feats_l = feat_extractor(feats_l.float())
    feats_r = feat_extractor(feats_r.float())
    print(f"feats_l 2 size: {feats_l.size()}")

    output_l,_ = model(feats_l)
    output_r,_ = model(feats_r)

    output = torch.maximum(output_l,output_r)
    print(f"predicted correctness: {output.item()}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_file", help="the pretrained model file", default = "snippets/"
    )

    parser.add_argument(
        "--debug", help="use a tiny dataset for debugging", default=False, action='store_true'
    )

    parser.add_argument(
        "--input_audio", help="the file containing input audio", default = "snippets/S08509/S08503_L0219_E009.wav"
    )

    args = parser.parse_args()
    
    main(args)
