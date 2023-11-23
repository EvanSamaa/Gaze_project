import sys
sys.path.append('..')
import torch
from torchmetrics.classification import BinaryF1Score, F1Score
import wandb
from tqdm import trange
import os
from datetime import datetime
from sklearn import metrics
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

sys.path.insert(0, '/Users/evanpan/Documents/GitHub/EvansToolBox/Utils')
sys.path.insert(0, '/Users/evanpan/Desktop/openpose/python/')
sys.path.insert(0, '/scratch/ondemand27/evanpan/EvansToolBox/Utils/')
sys.path.insert(0, '/scratch/ondemand27/evanpan/Gaze_project/')

# from training.model import *
from Dataset_Util.dataloader import *

class SentenceBaseline_GazePredictionModel(nn.Module):
    def __init__(self, config):
        torch.set_default_tensor_type(torch.DoubleTensor)
        # initialize model
        super(SentenceBaseline_GazePredictionModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation = nn.Sigmoid()
        self.num_layers = config["num_layers"]
        self.config = config
        # the feature of each speaker are encoded with a separate Linear Layer
        self.input_layer_self = nn.Linear(int(config["input_dims"]/2 - 6), config["input_layer_out"])
        self.input_layer_other = nn.Linear(int(config["input_dims"]/2 - 6), config["input_layer_out"])
        
        # the Recurrent Layer will take care of the next step
        self.lstm_hidden_dims = config["lstm_output_feature_size"]
        self.num_lstm_layer = config["lstm_layer_num"]
        self.frames_ahead = config["frames_ahead"]
        self.frames_behind = config["frames_behind"]
        self.lstm = nn.LSTM(2 * (config["input_layer_out"] + 6) * (self.frames_ahead + self.frames_behind + 1), 
                            self.lstm_hidden_dims, 
                            self.num_lstm_layer, 
                            batch_first=True,
                            bidirectional=True)     
        self.bn_lstm = nn.BatchNorm1d(self.lstm_hidden_dims * 2)   
        self.bn_layer_1 = nn.BatchNorm1d(config["output_layer_1_hidden"])
        self.bn_layer_2 = nn.BatchNorm1d(config["output_layer_2_hidden"])
        # output layers
        self.output_layer_1 = nn.Linear(self.lstm_hidden_dims * 2, config["output_layer_1_hidden"])
        self.output_layer_1 = nn.Sequential(self.output_layer_1, self.activation)
        self.output_layer_2 = nn.Linear(config["output_layer_1_hidden"], config["output_layer_2_hidden"])
        self.output_layer_2 = nn.Sequential(self.output_layer_2, self.activation)
        self.output_layer_3 = nn.Linear(config["output_layer_2_hidden"], config["output_layer_3_hidden"])
        self.output_layer_3 = nn.Sequential(self.output_layer_3)

        # audio_filler = torch.tensor([[[-36.04365338911715,0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715]]]).to(self.device)
        # text_filler = torch.ones([1, 1, 772]).to(self.device) * -15
        # text_filler[:, :, -4:] = 0
        # self.filler = torch.concat([audio_filler, text_filler], axis=2)
    def concate_frames(self, input_feature):
        # here I expect the 
        padding_front = torch.zeros([input_feature.shape[0], self.frames_ahead, input_feature.shape[2]]).to(self.device)
        padding_back = torch.zeros([input_feature.shape[0], self.frames_behind, input_feature.shape[2]]).to(self.device)
        padded_input_audio = torch.cat([padding_front, input_feature, padding_back], dim=1)
        window_audio = []
        for i in range(0, input_feature.shape[1]):
            window_count = i + self.frames_ahead
            current_window = padded_input_audio[:, window_count-self.frames_ahead:window_count+self.frames_behind+1]
            s = current_window.shape
            current_window = current_window.view((s[0], s[1] * s[2]))
            current_window = torch.unsqueeze(current_window, 1)
            window_audio.append(current_window)
        rtv = torch.cat(window_audio, dim=1)
        return rtv
    def forward(self, input_feature):
        feature_size = int(input_feature.size()[2] / 2)
        mod_audio_self = input_feature[:, :, :feature_size]
        mod_audio_other = input_feature[:, :, feature_size:]
        
        text_feature_self = mod_audio_self[:, :, :6]
        mod_audio_self = mod_audio_self[:, :, 6:]
        text_feature_other = mod_audio_other[:, :, :6]
        mod_audio_other = mod_audio_other[:, :, 6:]
        x1 = self.activation(self.input_layer_self(mod_audio_self))
        x2 = self.activation(self.input_layer_self(mod_audio_other))
        x1_windowed = self.concate_frames(x1)
        x2_windowed = self.concate_frames(x2)
        x_combined = torch.concat([x1_windowed, text_feature_self, x2_windowed, text_feature_other], axis=2)
        # here I'm assuming that the input_audio is of proper shape
        out, hidden_state = self.lstm(x_combined)
        # batch norm
        out = self.bn_lstm(out.permute(0, 2, 1)).permute(0, 2, 1)
        # x = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.activation(out)
        x = self.output_layer_1(x)
        # batch norm layer 1
        x = self.bn_layer_1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.output_layer_2(x)
        # batch norm layer 2
        x = self.bn_layer_2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.output_layer_3(x)
        return x
    def load_weights(self, pretrained_dict):
    #   not_copy = set(['fc.weight', 'fc.bias'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
def get_velocity(out):
    vel = torch.diff(out, dim=1, prepend=out[:, 0:1, :])
    return vel
def train_model_with_vel(model, config, train_data, valid_data, wandb, model_name, start = 1):
    optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train() 
    loss_fn = nn.CrossEntropyLoss()
    loss_fn_vel = nn.CrossEntropyLoss()
    training_loss = []
    valid_loss = []
    training_f1 = []
    valid_f1 = []
    aversion_vs_start = []
    count = 0
    # f1_score = BinaryF1Score(num_classes=2).to(device)
    f1_score = F1Score(task="multiclass", num_classes=2, average="weighted").to(device)
    for epoch in range(start, config['epochs'] + 1):
        total_train_loss = 0
        total_valid_loss = 0
        total_aversion_predicted = 0
        total_train_f1 = 0
        total_valid_f1 = 0
        train_batch_counter = 0
        valid_batch_counter = 0
        total_prediction_counter = 0
        prediction_mean = 0
        prediction_std = 0
        model.zero_grad()
        for _, (X, [Y, Y_vel]) in enumerate(train_data):
            model.train()
            train_batch_counter += 1
            X, Y, Y_vel = X.to(device), Y.to(device), Y_vel.to(device)            
            optimiser.zero_grad()
            if "Transformer" in config["model_type"]:
                all_zero = torch.zeros(Y.shape).to(device)
                pred = model(X, all_zero)
            else:
                pred = model(X)
            loss = loss_fn(pred.transpose(2, 1), Y.long())
            # get the softmax
            pred_vel = get_velocity(torch.softmax(pred, axis=2))
            # add the velocity along dimension 1 and 2 respectively, the velocity are opposite 
            loss = 0.001 * loss + loss_fn_vel(pred_vel[:, :, 0], -Y_vel) + loss_fn_vel(pred_vel[:, :, 1], Y_vel)
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item()
            # binary_pred = torch.round(pred)
            binary_pred = torch.argmax(pred, axis=2, keepdim=True)
            prediction_mean = torch.mean(binary_pred.float()).item()
            prediction_std = torch.std(binary_pred.float()).item()            
            f1_train = f1_score(binary_pred, torch.unsqueeze(Y, axis=2)).item()
            total_aversion_predicted += torch.sum(binary_pred).item()
            total_prediction_counter += binary_pred.size()[0] * binary_pred.size()[1] 
            total_train_f1 += f1_train
            del X, Y, pred
            torch.cuda.empty_cache()

        total_train_f1 /= train_batch_counter
        total_train_loss /= len(train_data)
        total_aversion_predicted /= total_prediction_counter

        for _, (X, [Y, Y_vel]) in enumerate(valid_data):
            with torch.no_grad():
                model.eval()
                valid_batch_counter += 1
                X, Y = X.to(device), Y.to(device)
                if "Transformer" in config["model_type"]:
                    all_zero = torch.zeros(Y.shape).to(device)
                    pred = model(X, all_zero)
                else:
                    pred = model(X)
                loss = loss_fn(pred.transpose(2, 1), Y.long())
                total_valid_loss += loss.item()

                # binary_pred = torch.round(pred)
                binary_pred = torch.argmax(pred, axis=2, keepdim=True)
                f1_valid = f1_score(binary_pred, torch.unsqueeze(Y, axis=2)).item()
                total_valid_f1 += f1_valid
                del X, Y, pred
                torch.cuda.empty_cache()

        total_valid_f1 /= valid_batch_counter
        total_valid_loss /= len(valid_data)

        if config['wandb']:
            wandb.log({'training loss': total_train_loss,
                        'validation_loss': total_valid_loss,
                        'training_f1': total_train_f1,
                        'validation_f1': total_valid_f1, 
                        "percentage_predicted_aversion": total_aversion_predicted})
        training_loss.append(total_train_loss)
        valid_loss.append(total_valid_loss)
        training_f1.append(total_train_f1)
        valid_f1.append(total_valid_f1)
        aversion_vs_start.append(total_aversion_predicted)
        if total_valid_f1 == max(valid_f1):
            try:
                os.mkdir(os.path.join(*[model_save_location, model_name]))
            except:
                pass
            config_save_path = os.path.join(*[model_save_location, model_name, "config.json"])
            json.dump(config, open(config_save_path, "w"))
            file_name = f'time={datetime.now()}_epoch={epoch}.pt'
            save_path = os.path.join(*[model_save_location, model_name, file_name])
            torch.save(model.state_dict(), save_path)
        if config['early_stopping']>0:
            if epoch > 1:
                if total_valid_f1 < np.mean(valid_f1[epoch - 7:epoch - 2]):
                    count += 1
                else:
                    count = 0
            if count >= config['early_stopping']:
                print('\n\nStopping early due to decrease in performance on validation set\n\n')
                break 
        if count == 0:
            print("Epoch {}, mean: {}, std: {}\ntraining L: {}\nvalidation L:{}".format(epoch, prediction_mean, prediction_std, total_train_f1, total_valid_f1))
        else:
            print("Epoch {}, mean: {}, std: {}\ntraining L: {}\nvalidation L:{}, model have not improved for {} iterations".format(epoch, prediction_mean, prediction_std, total_train_f1, total_valid_f1, count))
    if config['wandb']:
        save_path = os.path.join(*[model_save_location, model_name, file_name])
        wandb.save(save_path)


if __name__ == "__main__":


    # inputs
    dataset_location = "/scratch/ondemand27/evanpan/data/deep_learning_processed_dataset"
    model_save_location = "/scratch/ondemand27/evanpan/data/Gaze_aversion_models"
    config = json.load(open("/scratch/ondemand27/evanpan/Gaze_project/training/sentence_config.json", "r"))
    
    config["load_model"] = False
    config["wandb"] = True
    config["batch_size"] = 32


    # do the training test split here:
    dataset_metadata = "/scratch/ondemand27/evanpan/data/deep_learning_processed_dataset/video_to_window_metadata.json"
    dataset_metadata = json.load(open(dataset_metadata, "r"))
    all_videos = list(dataset_metadata.keys())
    training_set = []
    testing_set = []
    # get the name of the videos (this ensures no contamination because the same shot is split)
    for i in range(0, len(all_videos)):
        if i / len(all_videos) < 0.9:
            training_set.append(all_videos[i])
        else:
            testing_set.append(all_videos[i])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    training_dataset = Aversion_SelfTap111(dataset_location, training_set, sentence_and_word_timing=True, velocity_label=True)
    validation_dataset = Aversion_SelfTap111(dataset_location, testing_set, sentence_and_word_timing=True, velocity_label=True)
    train_dataloader = torch.utils.data.DataLoader(training_dataset, config['batch_size'], True)
    valid_dataloader = torch.utils.data.DataLoader(validation_dataset, config['batch_size'], True)

    model = SentenceBaseline_GazePredictionModel(config)

    run_obj = None
    if config["wandb"]:
        wandb.login()
        run_obj = wandb.init(project="gaze_prediction", config=config, settings=wandb.Settings(start_method="fork"))

    train_model_with_vel(model, config, train_dataloader, valid_dataloader, run_obj, "sentence_and_words_with_vel_2023_Oct_prevent_overfitting_smaller_model_vel_only")
    run_obj.finish()
