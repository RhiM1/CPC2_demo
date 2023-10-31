from torch import nn
import torch
from transformers import  WhisperFeatureExtractor, WhisperForConditionalGeneration


class WhisperWrapper_full(nn.Module):
    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None, num_layers = 12, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # using layer = -1 returns all layers in form (1, time, feat_dim, layers)
        # otherwise single layer in form (1, time, feat_dim)

        self.num_layers = num_layers
        self.use_feat_extractor = use_feat_extractor
        if layer is None:
            self.layer = 12
        else:
            self.layer = layer

        if use_feat_extractor:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        if pretrained_model is None:
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(pretrained_model)

        
    def forward(self, data):

        if self.use_feat_extractor:
            print(f"data size 1: {data.size()}")
            data = self.feature_extractor(data, sampling_rate = 16000, return_tensors = 'pt')

        print(f"{data}")
        outputs = self.model.generate(
            input_features = data["input_features"],
            output_hidden_states = True,
            return_dict_in_generate = True
        )

        if self.layer == -1:
            decoder_hidden = []
            for layer in range(self.num_layers):
                hidden = torch.stack([outputs.decoder_hidden_states[word][layer][0][0] for word in range(len(outputs.decoder_hidden_states))])
                decoder_hidden.append(hidden.unsqueeze(0))
            decoder_hidden = torch.stack(decoder_hidden, dim = -1)
        else:
            decoder_hidden = torch.stack([outputs.decoder_hidden_states[word][self.layer][0][0] for word in range(len(outputs.decoder_hidden_states))])
            decoder_hidden = decoder_hidden.unsqueeze(0)

        return decoder_hidden

