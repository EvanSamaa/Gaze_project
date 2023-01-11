import torch.nn as nn
import torch
import torch.functional as F
import torchaudio
import parselmouth
import yaml

class CNNet(nn.Module):
    def __init__(self, config, single_datapoint_shape=(1,40,1201), target_shape=50):
        super().__init__()
        self.conv_layers = []
        conv_layers = config['conv_layers']
        kernel_size = config['kernel_size']
        padding = config['padding']
        for i in range(len(conv_layers)):
            if i < len(conv_layers) - 1:
                in_ = conv_layers[i]
                out_ = conv_layers[i+1]
                self.conv_layers.append(nn.Conv2d(in_, out_, kernel_size=kernel_size, padding=padding))
                if config['pool'] is not None:
                    self.conv_layers.append(config['pool'])
                self.conv_layers.append(nn.BatchNorm2d(out_))
                self.conv_layers.append(config['activation_fn'])

        self.conv = nn.Sequential(*self.conv_layers)

        fake_data = torch.ones(single_datapoint_shape)
        fake_data = fake_data.unsqueeze(1)
        fake_data = self.conv(fake_data)

        fc_input = torch.prod(torch.tensor(fake_data.shape))
        self.fc1 = nn.Linear(fc_input, fc_input//4)
        self.fc2 = nn.Linear(fc_input//4, target_shape)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
    
    def load_weights(self, pretrained_dict):
    #   not_copy = set(['fc.weight', 'fc.bias'])
        not_copy = set()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in not_copy}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

def normalise_tensor(matrix):
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
def load_single_audio_file_normalised(file_path, time_step=0.1):
    waveform, sample_rate = torchaudio.load(file_path)
    mfcc_spectogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
    # Intensity
    snd = parselmouth.Sound(file_path)
    intensity = torch.tensor(snd.to_intensity(time_step=time_step).values).flatten()
    to_pad = mfcc_spectogram.shape[2] - intensity.shape[0]
    intensity = torch.cat([intensity, torch.zeros(to_pad)], 0).to(torch.float32)
    mfcc_spectogram = torch.cat([mfcc_spectogram, intensity.unsqueeze(0).unsqueeze(0)], 1)
    # return mfcc_spectogram
    return normalise_tensor(mfcc_spectogram.squeeze(0))


def predict_aversion(audio_file_path):
    config = {
        "conv_layers": [1, 8, 32, 64, 128, 256],
        "kernel_size": 5,
        "padding": 2,
        "activation_fn": nn.ReLU(),
        "pool": nn.AvgPool2d(2),
        # "pool": None,
        "epochs": 100,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        # "loss_fn": nn.MultiLabelSoftMarginLoss(),
        # "loss_fn": nn.BCELoss(),
        # "loss_fn": utils.weighted_binary_cross_entropy,
        "learning_rate": 0.00001,
        "batch_size": 64,
        "wandb": True,
        "load_model": False,
        "early_stopping": 5,
        "window_length": 0.1,
        "time_step": 0.01 # window_length / this shoulkd be <= 1201 (for 5 sec samples)
    }
    checkpoint_path = '../../data/ribhav_model/time=2022-11-29 06_22_13.018369_epoch=10.pt'
    config_path = "../../data/ribhav_model/config.yaml"
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    for key in yaml_config:
        try:
            if 'value' in yaml_config[key]:
                config[key] = yaml_config[key]['value']
        except:
            pass
    config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    # activation_fn
    if 'ReLU' in config["activation_fn"]:
        config["activation_fn"] = nn.ReLU()
    # pool
    start_ind = config['pool'].index('=') + 1
    end_ind = config['pool'].index(',')
    k_size = int(config['pool'][start_ind:end_ind])
    if 'AvgPool2d' in config['pool']:
        config['pool'] = nn.AvgPool2d(k_size)
    # loss_fn
    '''
    Finished Loading config
    '''

    x = load_single_audio_file_normalised(audio_file_path)
    x = x.to(config['device'])
    model = CNNet(config, [1] + list(x.shape), int(5/config["window_length"]))
    pretrained_dict = torch.load(checkpoint_path, map_location=config['device'])
    model.load_weights(pretrained_dict)
    model.to(config['device'])

    pred = model(x.unsqueeze(0))
    pred = pred.cpu().detach().numpy().flatten().tolist()
    print(len(pred))

if __name__ == "__main__":
    predict_aversion("/Users/evanpan/Desktop/_Number_0_channel_0_DVA2C.wav")