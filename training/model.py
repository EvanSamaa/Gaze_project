import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
class SimpleBaseline_GazePredictionModel(nn.Module):
    def __init__(self, config):
        # initialize model
        super(SimpleBaseline_GazePredictionModel, self).__init__()
        self.activation = nn.LeakyReLU()
        self.num_layers = config["num_layers"]
        # the feature of each speaker are encoded with a separate Linear Layer
        self.input_layer_self = nn.Linear(int(config["input_dims"]/2), config["input_layer_out"])
        self.input_layer_other = nn.Linear(int(config["input_dims"]/2), config["input_layer_out"])
        
        # the Recurrent Layer will take care of the next step
        lstm_hidden_dims = config["lstm_output_feature_size"]
        num_lstm_layer = config["lstm_layer_num"]
        self.frames_ahead = config["frames_ahead"]
        self.frames_behind = config["frames_behind"]
        self.lstm = nn.LSTM(2 * config["input_layer_out"] * (self.frames_ahead + self.frames_behind + 1), 
                            lstm_hidden_dims, 
                            num_lstm_layer, 
                            batch_first=True)
        
        # output layers
        self.output_layer_1 = nn.Linear(lstm_hidden_dims, config["output_layer_1_hidden"])
        self.output_layer_2 = nn.Linear(config["output_layer_1_hidden"], config["output_layer_2_hidden"])
        self.output_layer_3 = nn.Linear(config["output_layer_2_hidden"], config["output_layer_3_hidden"])

        self.audio_filler = torch.tensor([[[-36.04365338911715,0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715]]]).to(self.config["device"])
        self.text_filler = torch.ones([1, 1, 772]).to(self.config["device"]) * -15
        self.text_filler[:, :, -4:] = 0
        
    def concate_frames(self, input_feature):
        # here I expect the 
        padding_front = torch.zeros((input_feature.shape[0], self.frames_ahead, input_feature.shape[2])).to(self.config["device"])
        padding_back = torch.zeros((input_feature.shape[0], self.frames_behind, input_feature.shape[2])).to(self.config["device"])
        padded_input_audio = torch.cat([padding_front, input_feature, padding_back], dim=1)
        window_audio = []
        for i in range(0, input_feature.shape[1]):
            window_count = i + 12
            current_window = padded_input_audio[:, window_count-self.frames_ahead:window_count+self.frames_behind+1]
            s = current_window.shape
            current_window = current_window.view((s[0], s[1] * s[2]))
            current_window = torch.unsqueeze(current_window, 1)
            window_audio.append(current_window)
        rtv = torch.cat(window_audio, dim=1)
        return rtv
    def forward(self, input_feature):
        mod_audio = self.concate_frames(input_audio)
        # here I'm assuming that the input_audio is of proper shape
        hidden_state = [torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)).to(self.config["device"]), 
        torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)).to(self.config["device"])]
        out, hidden_state = self.lstm(mod_audio, hidden_state)
        # bn
        # x = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(out)
        x = self.output_mat1(x)
        x = x[:, :, 0]
        x = self.sigmoid(x)
        return x

        


