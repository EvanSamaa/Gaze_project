import numpy as np
import whisper_timestamped
import librosa
import python_speech_features as psf
import os
import json
import sys
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.special import softmax
import math
import shutil
import soundfile as sf
sys.path.insert(0, '/Users/evanpan/Documents/GitHub/EvansToolBox/Utils')
sys.path.insert(0, '/Users/evanpan/Desktop/openpose/python/')
sys.path.insert(0, '/scratch/ondemand27/evanpan/EvansToolBox/Utils/')
sys.path.insert(0, '/scratch/ondemand27/evanpan/Gaze_project/')
from Signal_processing_utils import intensity_from_signal

def find_index(lst, t):
    for i in range(0, len(lst)):
        if t >= lst[i][0] and t <= lst[i][1]:
            return i, lst[i]
        elif t <= lst[i][0]:
            if i > 0:
                return -1, [lst[i-1][1], lst[i][0]]
            else:
                return -1, [-1, lst[i][0]] # assume the silence started before the current frame
        if i == len(lst) - 1:
            return -1, [lst[i][1], lst[i][1]+1]  # assume the silence ends after the current frame
def generate_word_structure_values(ts, word_list):
    if len(word_list) == 0:
        output_vector = np.zeros([ts.shape[0], 6])
        return output_vector
    # generate word embedding per word
    sentence_words = []
    word_intervals = []
    for i in range(0, len(word_list)):
        sentence_words.append(word_list[i]["text"])
        word_intervals.append([word_list[i]["start"], word_list[i]["end"]])
    output_vector = np.zeros([ts.shape[0], 6])
    # features
    # time since start of word (0 in silence, which means it will start at zero)
    # time till end of word (0 in silence as well,)
    # length of sentence 
    # time since start of silence (0 when speaking)
    # time till end of silence (0 when speaking)
    # length of silence
    for i in range(0, ts.shape[0]):
        t = ts[i]
        index, [start, end] = find_index(word_intervals, t)
        if index == -1:
            output_vector[i, 3] = t - start
            output_vector[i, 4] = t - end
            output_vector[i, 5] = start - end
        else:
            output_vector[i, 0] = t - start
            output_vector[i, 1] = t - end
            output_vector[i, 2] = start - end
    return output_vector
def generate_sentence_structure_values_real(ts, sentence_interval):
    if len(sentence_interval) == 0:
        output_vector = np.zeros([ts.shape[0], 6])
        return output_vector
    output_vector = np.zeros([ts.shape[0], 6])
    # features
    # time since start of word (0 in silence, which means it will start at zero)
    # time till end of word (0 in silence as well,)
    # length of sentence 
    # time since start of silence (0 when speaking)
    # time till end of silence (0 when speaking)
    # length of silence
    for i in range(0, ts.shape[0]):
        t = ts[i]
        index, [start, end] = find_index(sentence_interval, t)
        if index == -1:
            output_vector[i, 3] = t - start
            output_vector[i, 4] = t - end
            output_vector[i, 5] = start - end
        else:
            output_vector[i, 0] = t - start
            output_vector[i, 1] = t - end
            output_vector[i, 2] = start - end
    return output_vector
def parse_for_sentence_intervals(transcript, threshold=0.56):
    if len(transcript) == 0:
        return [], []
    sentence_intervals = [] # this is used to only store [[start, end]]
    sentence_words = [] # this is used to store [[all words of sentence]]
    current_sentence = [transcript[0]]
    for i in range(1, len(transcript)):
        word = transcript[i]["text"]
        start = transcript[i]["start"]
        end = transcript[i]["end"]
        if start - current_sentence[-1]["end"] >= threshold:
            sentence_words.append(current_sentence)
            sentence_intervals.append([current_sentence[0]["start"], current_sentence[-1]["end"]])
            current_sentence = [transcript[i]]
        else:
            current_sentence.append(transcript[i])
    return sentence_intervals, sentence_words
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
                            batch_first=True)        
        # output layers
        self.output_layer_1 = nn.Linear(self.lstm_hidden_dims, config["output_layer_1_hidden"])
        self.output_layer_1 = nn.Sequential(self.output_layer_1, self.activation, nn.Dropout(self.config["dropout"]))
        self.output_layer_2 = nn.Linear(config["output_layer_1_hidden"], config["output_layer_2_hidden"])
        self.output_layer_2 = nn.Sequential(self.output_layer_2, self.activation, nn.Dropout(self.config["dropout"]))
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
        text_feature_other = mod_audio_self[:, :, :6]
        mod_audio_other = mod_audio_other[:, :, 6:]
        x1 = self.activation(self.input_layer_self(mod_audio_self))
        x2 = self.activation(self.input_layer_self(mod_audio_other))
        x1_windowed = self.concate_frames(x1)
        x2_windowed = self.concate_frames(x2)
        x_combined = torch.concat([x1_windowed, text_feature_self, x2_windowed, text_feature_other], axis=2)
        # here I'm assuming that the input_audio is of proper shape
        out, hidden_state = self.lstm(x_combined)
        # bn
        # x = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.activation(out)
        x = self.output_layer_1(x)
        x = self.output_layer_2(x)
        x = self.output_layer_3(x)
        return x
    def load_weights(self, pretrained_dict):
    #   not_copy = set(['fc.weight', 'fc.bias'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)  
def merge_transcript(list1, list2):
    i = 0
    j = 0
    result = ""
    speaker = -1
    while i < len(list1) and j < len(list2):
        if list1[i]["start"] <= list2[j]["start"]:
            if speaker != 1:
                result += "\nspeaker_0: "
                speaker = 1
            result += str(list1[i]["text"]) + " "
            i += 1
        else:
            if speaker != 2:
                result += "\nspeaker_1: "
                speaker = 2
            result += str(list2[j]["text"]) + " "
            j += 1
    while i < len(list1 ):
        if speaker != 1:
            result += "\nspeaker_0: "
            speaker = 1
        result += str(list1[i]["text"]) + " "
        i += 1
    while j < len(list2):
        if speaker != 2:
            result += "\nspeaker_1: "
            speaker = 2
        result += str(list2[j]["text"]) + " "
        j += 1
    return result.strip()

class Aversion111Prior():
    def __init__(self, model_location="/scratch/ondemand27/evanpan/data/Gaze_aversion_models/sentence_and_words",
                 whisper_root = "/scratch/ondemand27/evanpan/data/deep_learning_processed_dataset") -> None:
        self.whisper_root = whisper_root
        self.model_location = model_location
        config = json.load(open(os.path.join(*[self.model_location, "config.json"]), "r"))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_weights_path = os.path.join(*[self.model_location, "best.pt"])
        pretrained_dict = torch.load(model_weights_path, map_location=self.device)
        self.model = SentenceBaseline_GazePredictionModel(config)
        # get the raw transcript prediction
        self.model.load_weights(pretrained_dict)
        self.model.to(self.device)

        torch.set_default_tensor_type(torch.FloatTensor)
        self.model_word = whisper_timestamped.load_model("base.en", download_root=os.path.join(self.whisper_root, "whisper"))
        torch.set_default_tensor_type(torch.DoubleTensor)
    def divide_and_conquer(self, X, segment_length):
        overlap = segment_length/2
        num_segments = math.ceil(X.size()[1] / segment_length)
        out = []
        for i in range(0, num_segments):
            end = int(np.minimum(X.size()[1], (i+1) * segment_length+overlap))
            X_seg = X[:, i * segment_length:end]
            out.append(self.model(X_seg)[0].cpu().detach().numpy())
        end = int(np.minimum(X.size()[1], segment_length))
        out_vec = out[0][:end]
        for i in range(1, len(out)):      
            end = int(np.minimum(out[i].shape[0], segment_length))
            out_vec = np.concatenate([out_vec, out[i][:end]])
        return out_vec
    def predict(self, temp_folder, input_folder, input_file_name, speaker, divide_and_conquer=False, segment_length=500, in_dataset=True):
        target_fps = 25
        no_space_input_file_name = input_file_name.replace(" ", "_")
        # get input paths
        if in_dataset:
            audio_path_self = os.path.join(*[input_folder, "audio", input_file_name+"_{}.wav".format(speaker)])
            audio_path_other = os.path.join(*[input_folder, "audio", input_file_name+"_{}.wav".format(1-speaker)])
        else:
            audio_path_self = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.wav".format(speaker)])
            audio_path_other = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.wav".format(1-speaker)])
        # dialog output path
        dialog_output_path = os.path.join(*[temp_folder, input_file_name+"_dialog_transcript.txt"])
        tagged_dialog_output_path_0 = os.path.join(*[temp_folder, input_file_name+"_dialog_transcript_tagging_0.txt"])
        tagged_dialog_output_path_1 = os.path.join(*[temp_folder, input_file_name+"_dialog_transcript_tagging_1.txt"])
        
        # output paths
        word_output_path = os.path.join(*[temp_folder, input_file_name+"_transcript.json"])
        processed_input_vector_self_path = os.path.join(*[temp_folder, input_file_name+"_input_feature_self.npy"])
        processed_input_vector_other_path = os.path.join(*[temp_folder, input_file_name+"_input_feature_other.npy"])
        # output paths for JALI
        self_script_output_path = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.txt".format(speaker)])
        other_script_output_path = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.txt".format(1-speaker)])
        self_script_tagged_output_path = os.path.join(*[temp_folder, no_space_input_file_name+"_{}_tagged.txt".format(speaker)])
        other_script_tagged_output_path = os.path.join(*[temp_folder, no_space_input_file_name+"_{}_tagged.txt".format(1-speaker)])
        self_audio_output_path = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.wav".format(speaker)])
        other_audio_output_path = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.wav".format(1-speaker)])
        if not os.path.exists(processed_input_vector_other_path):
            torch.set_default_tensor_type(torch.FloatTensor)
            # if the input was not generated
            if not os.path.exists(word_output_path):
                raw_word_predictions_self = whisper_timestamped.transcribe(self.model_word, audio_path_self, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True)
                raw_word_predictions_other = whisper_timestamped.transcribe(self.model_word, audio_path_other, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True)
                # generate the json file containing word timing
                word_alignment_self = []
                for s in range(0,len(raw_word_predictions_self["segments"])):
                    word_alignment_self = word_alignment_self + raw_word_predictions_self["segments"][s]["words"]
                word_alignment_other = []
                for s in range(0,len(raw_word_predictions_other["segments"])):
                    word_alignment_other = word_alignment_other + raw_word_predictions_other["segments"][s]["words"]
                trascript_json = {"self":word_alignment_self, "other":word_alignment_other}
                json.dump(trascript_json, open(word_output_path, "w"))
            else:
                with open(word_output_path, "rb") as f:
                    trascript_json = json.load(f)
            # get audio information
            audio_self, sr = librosa.load(audio_path_self)
            audio_other, sr = librosa.load(audio_path_other)
            max_length = np.maximum(audio_self.shape[0], audio_other.shape[0])
            audio_self = np.concatenate([audio_self, np.zeros([max_length - audio_self.shape[0], ])], axis=0)
            audio_other = np.concatenate([audio_other, np.zeros([max_length - audio_other.shape[0], ])], axis=0)
            # padding these so we don't lose any frames (we are padding 
            # half a window of the windowed functionss)
            audio_self = np.pad(audio_self, ((int(0.02 * sr), int(0.02 * sr))))
            audio_other = np.pad(audio_other, ((int(0.02 * sr), int(0.02 * sr))))
            # generate the audio features for speaker 2
            mfcc_a0 = psf.mfcc(audio_self, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
            logfbank_a0 = psf.logfbank(audio_self, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            ssc_feat_a0 = psf.logfbank(audio_self, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            full_feat_self = np.concatenate([mfcc_a0, logfbank_a0, ssc_feat_a0], axis=1)
            # generate the audio features for speaker 1
            mfcc_a1 = psf.mfcc(audio_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
            logfbank_a1 = psf.logfbank(audio_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            ssc_feat_a1 = psf.logfbank(audio_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            full_feat_other = np.concatenate([mfcc_a1, logfbank_a1, ssc_feat_a1], axis=1)

            words = json.load(open(word_output_path, "r"))
            self_text = words["self"]
            other_text = words["other"]

            ts_target = np.arange(0, full_feat_other.shape[0]) / target_fps
            # get word vector
            word_structure_feature_self = generate_word_structure_values(ts_target, self_text)
            word_structure_feature_other = generate_word_structure_values(ts_target, other_text)
            # get the sentence vector
            interval_of_sentences_self, ___ = parse_for_sentence_intervals(self_text, 0.56)
            interval_of_sentences_other, ___ = parse_for_sentence_intervals(other_text, 0.56)
            sentence_structure_feature_self = generate_sentence_structure_values_real(ts_target, interval_of_sentences_self)
            sentence_structure_feature_other = generate_sentence_structure_values_real(ts_target, interval_of_sentences_other)

            output_feature_just_self = np.concatenate([full_feat_self, word_structure_feature_self, sentence_structure_feature_self], axis=1)
            output_feature_just_other = np.concatenate([full_feat_other, word_structure_feature_other, sentence_structure_feature_other], axis=1)
            output_feature_self = np.concatenate([output_feature_just_self, output_feature_just_other], axis=1)
            output_feature_other = np.concatenate([output_feature_just_other, output_feature_just_self], axis=1)
            np.save(processed_input_vector_self_path, output_feature_self)
            np.save(processed_input_vector_other_path, output_feature_other)
        # also output the file into the temp folder for JALI to generate the lip motion
        if not os.path.exists(self_script_output_path):
            script = json.load(open(word_output_path, "r"))
            self_script = script["self"]
            # get the text of other
            self_out_script = ""
            for i in range(0, len(self_script)):
                self_out_script += " " + self_script[i]["text"]
            # get the text of self
            other_script = script["other"]
            other_out_script = ""
            for i in range(0, len(other_script)):
                other_out_script += " " + other_script[i]["text"]
            with open(self_script_output_path, "w") as file:
                file.write(self_out_script)
            with open(other_script_output_path, "w") as file:
                file.write(other_out_script)
            if not in_dataset:
                # shutil.copy2(audio_path_self, self_audio_output_path)
                # shutil.copy2(audio_path_other, other_audio_output_path)
                pass            
            shutil.copy2(self_script_output_path, self_script_tagged_output_path)
            shutil.copy2(other_script_output_path, other_script_tagged_output_path)
        if not os.path.exists(dialog_output_path):
            with open(word_output_path, "r") as f:
                transcribe_json = json.load(f)
            speaker_0 = transcribe_json["self"]
            speaker_1 = transcribe_json["other"]
            output_string = merge_transcript(speaker_0, speaker_1)
            with open(dialog_output_path, "w") as f:
                f.write(output_string)
            with open(tagged_dialog_output_path_0, "w") as f:
                f.write(output_string)
            with open(tagged_dialog_output_path_1, "w") as f:
                f.write(output_string)
        torch.set_default_tensor_type(torch.DoubleTensor)
        if speaker == 0:
            X = torch.from_numpy(np.expand_dims(np.load(processed_input_vector_self_path), axis=0)).to(self.device)
        else:
            X = torch.from_numpy(np.expand_dims(np.load(processed_input_vector_other_path), axis=0)).to(self.device)
        if divide_and_conquer:
            out = self.divide_and_conquer(X, segment_length)
        else:
            out = self.model(X)[0].cpu().detach().numpy()
        out = softmax(out, axis=1)
        return out, X



class Aversion111Prior_three_party():
    def __init__(self, model_location="/scratch/ondemand27/evanpan/data/Gaze_aversion_models/sentence_and_words",
                 whisper_root = "/scratch/ondemand27/evanpan/data/deep_learning_processed_dataset") -> None:
        self.whisper_root = whisper_root
        self.model_location = model_location
        config = json.load(open(os.path.join(*[self.model_location, "config.json"]), "r"))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_weights_path = os.path.join(*[self.model_location, "best.pt"])
        pretrained_dict = torch.load(model_weights_path, map_location=self.device)
        self.model = SentenceBaseline_GazePredictionModel(config)
        # get the raw transcript prediction
        self.model.load_weights(pretrained_dict)
        self.model.to(self.device)

        torch.set_default_tensor_type(torch.FloatTensor)
        self.model_word = whisper_timestamped.load_model("base.en", download_root=os.path.join(self.whisper_root, "whisper"))
        torch.set_default_tensor_type(torch.DoubleTensor)
    def predict(self, temp_folder, input_folder, input_file_name, speaker, divide_and_conquer=False, segment_length=500, in_dataset=True):
        target_fps = 25
        no_space_input_file_name = input_file_name.replace(" ", "_")
        # get input paths (here we have 3 paths, and we will generate aversion probability for all three of them)
        if in_dataset:
            # this will never happen
            audio_path_0 = os.path.join(*[input_folder, "audio", input_file_name+"_{}.wav".format(0)])
            audio_path_1 = os.path.join(*[input_folder, "audio", input_file_name+"_{}.wav".format(1)])
            audio_path_2 = os.path.join(*[input_folder, "audio", input_file_name+"_{}.wav".format(2)])
        else:
            # take in three pathes
            audio_path_0 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.wav".format(0)])
            audio_path_1 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.wav".format(1)])
            audio_path_2 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.wav".format(2)])
            
            audio_path_0_other = os.path.join(*[temp_folder, no_space_input_file_name+"_{}_other.wav".format(0)])
            audio_path_1_other = os.path.join(*[temp_folder, no_space_input_file_name+"_{}_other.wav".format(1)])
            audio_path_2_other = os.path.join(*[temp_folder, no_space_input_file_name+"_{}_other.wav".format(2)])
        
            # check if audio_path_0_other exists
            if not os.path.exists(audio_path_0_other) or True:
                # generate these files by combining the other two
                audio_0, sr1 = sf.read(audio_path_0)
                audio_1, sr = sf.read(audio_path_1)
                audio_2, sr = sf.read(audio_path_2)
                max_length = np.maximum(audio_0.shape[0], audio_1.shape[0])
                max_length = np.maximum(max_length, audio_2.shape[0])
                
                # normalize the length so they are all the same length
                audio_0 = np.concatenate([audio_0, np.zeros([max_length - audio_0.shape[0], audio_0.shape[1]])], axis=0)
                audio_1 = np.concatenate([audio_1, np.zeros([max_length - audio_1.shape[0], audio_1.shape[1]])], axis=0)
                audio_2 = np.concatenate([audio_2, np.zeros([max_length - audio_2.shape[0], audio_2.shape[1]])], axis=0)
                audio_0_other = audio_1 + audio_2
                audio_1_other = audio_0 + audio_2
                audio_2_other = audio_0 + audio_1
                sf.write(audio_path_0_other, audio_0_other, sr1)
                sf.write(audio_path_1_other, audio_1_other, sr1)
                sf.write(audio_path_2_other, audio_2_other, sr1)

                # for good measures, save the audio files with the normalized lengths
                sf.write(audio_path_0, audio_0, sr1)
                sf.write(audio_path_1, audio_1, sr1)
                sf.write(audio_path_2, audio_2, sr1)
        # also make preparations to generate other pre-computaiton files
        # each dialog transcript stores the the 
        tagged_dialog_output_path_0 = os.path.join(*[temp_folder, input_file_name+"_dialog_transcript_tagging_0.txt"])
        tagged_dialog_output_path_1 = os.path.join(*[temp_folder, input_file_name+"_dialog_transcript_tagging_1.txt"])
        tagged_dialog_output_path_2 = os.path.join(*[temp_folder, input_file_name+"_dialog_transcript_tagging_2.txt"])
        
        # output paths
        word_output_path = os.path.join(*[temp_folder, input_file_name+"_transcript.json"])
        processed_input_vector_0_path = os.path.join(*[temp_folder, input_file_name+"_input_feature_0.npy"])
        processed_input_vector_1_path = os.path.join(*[temp_folder, input_file_name+"_input_feature_1.npy"])
        processed_input_vector_2_path = os.path.join(*[temp_folder, input_file_name+"_input_feature_2.npy"])
        # output paths for JALI
        script_output_path_0 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.txt".format(0)])
        script_output_path_1 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.txt".format(1)])
        script_output_path_2 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}.txt".format(2)])
        script_tagged_output_path_0 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}_tagged.txt".format(0)])
        script_tagged_output_path_1 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}_tagged.txt".format(1)])
        script_tagged_output_path_2 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}_tagged.txt".format(2)])
        
        praat_path_0 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}_PraatOutput.txt".format(0)])
        praat_path_1 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}_PraatOutput.txt".format(1)])
        praat_path_2 = os.path.join(*[temp_folder, no_space_input_file_name+"_{}_PraatOutput.txt".format(2)])
        if os.path.exists(praat_path_0):
            # generate the word_alignment files using the praat file
            _, _, words_0, word_interval_0 = generate_word_alignment(praat_path_0)
            _, _, words_1, word_interval_1 = generate_word_alignment(praat_path_1)
            _, _, words_2, word_interval_2 = generate_word_alignment(praat_path_2)
            
            word_alignment_0 = generate_transcript_whisper_style(words_0, word_interval_0)
            word_alignment_1 = generate_transcript_whisper_style(words_1, word_interval_1)
            word_alignment_2 = generate_transcript_whisper_style(words_2, word_interval_2)
            # generate other_transcript 
            word_alignment_0_other = combine_whisper_transcripts(word_alignment_1, word_alignment_2)
            word_alignment_1_other = combine_whisper_transcripts(word_alignment_0, word_alignment_2)
            word_alignment_2_other = combine_whisper_transcripts(word_alignment_0, word_alignment_1)
            trascript_json = {"0":word_alignment_0, "1":word_alignment_1, "2":word_alignment_2, "0_other":word_alignment_0_other, "1_other":word_alignment_1_other, "2_other":word_alignment_2_other}
            json.dump(trascript_json, open(word_output_path, "w"))
            
        
        if not os.path.exists(processed_input_vector_0_path):
            
            torch.set_default_tensor_type(torch.FloatTensor)
            # if the input was not generated
            if not os.path.exists(word_output_path):
                raw_word_predictions_0 = whisper_timestamped.transcribe(self.model_word, audio_path_0, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True)
                raw_word_predictions_1 = whisper_timestamped.transcribe(self.model_word, audio_path_1, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True)
                raw_word_predictions_2 = whisper_timestamped.transcribe(self.model_word, audio_path_2, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True)
                raw_word_predictions_0_other = whisper_timestamped.transcribe(self.model_word, audio_path_0_other, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True)
                raw_word_predictions_1_other = whisper_timestamped.transcribe(self.model_word, audio_path_1_other, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True)
                raw_word_predictions_2_other = whisper_timestamped.transcribe(self.model_word, audio_path_2_other, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True)
                # generate the json file containing word timing
                word_alignment_0 = []
                for s in range(0,len(raw_word_predictions_0["segments"])):
                    word_alignment_0 = word_alignment_0 + raw_word_predictions_0["segments"][s]["words"]
                word_alignment_1 = []
                for s in range(0,len(raw_word_predictions_1["segments"])):
                    word_alignment_1 = word_alignment_1 + raw_word_predictions_1["segments"][s]["words"]
                word_alignment_2 = []
                for s in range(0,len(raw_word_predictions_2["segments"])):
                    word_alignment_2 = word_alignment_2 + raw_word_predictions_2["segments"][s]["words"]
                    
                word_alignment_0_other = []
                for s in range(0,len(raw_word_predictions_0_other["segments"])):
                    word_alignment_0_other = word_alignment_0_other + raw_word_predictions_0_other["segments"][s]["words"]
                word_alignment_1_other = []
                for s in range(0,len(raw_word_predictions_1_other["segments"])):
                    word_alignment_1_other = word_alignment_1_other + raw_word_predictions_1_other["segments"][s]["words"]
                word_alignment_2_other = []
                for s in range(0,len(raw_word_predictions_2_other["segments"])):
                    word_alignment_2_other = word_alignment_2_other + raw_word_predictions_2_other["segments"][s]["words"]
                    
                trascript_json = {"0":word_alignment_0, "1":word_alignment_1, "2":word_alignment_2, "0_other":word_alignment_0_other, "1_other":word_alignment_1_other, "2_other":word_alignment_2_other}                    
                json.dump(trascript_json, open(word_output_path, "w"))
            else:
                with open(word_output_path, "rb") as f:
                    trascript_json = json.load(f)
            
            audio_0, sr = librosa.load(audio_path_0)
            audio_1, sr = librosa.load(audio_path_1)
            audio_2, sr = librosa.load(audio_path_2)
            audio_0_other, sr = librosa.load(audio_path_0_other)
            audio_1_other, sr = librosa.load(audio_path_1_other)
            audio_2_other, sr = librosa.load(audio_path_2_other)
            
            # padding these so we don't lose any frames (we are padding
            # half a window of the windowed functionss)
            audio_0 = np.pad(audio_0, ((int(0.02 * sr), int(0.02 * sr))))
            audio_1 = np.pad(audio_1, ((int(0.02 * sr), int(0.02 * sr))))
            audio_2 = np.pad(audio_2, ((int(0.02 * sr), int(0.02 * sr))))
            
            audio_0_other = np.pad(audio_0_other, ((int(0.02 * sr), int(0.02 * sr))))
            audio_1_other = np.pad(audio_1_other, ((int(0.02 * sr), int(0.02 * sr))))
            audio_2_other = np.pad(audio_2_other, ((int(0.02 * sr), int(0.02 * sr))))
            
            # generate the audio features for speaker 0
            mfcc_a0 = psf.mfcc(audio_0, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
            logfbank_a0 = psf.logfbank(audio_0, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            ssc_feat_a0 = psf.logfbank(audio_0, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            full_feat_0 = np.concatenate([mfcc_a0, logfbank_a0, ssc_feat_a0], axis=1)
            # generate the audio features for speaker 1
            mfcc_a1 = psf.mfcc(audio_1, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
            logfbank_a1 = psf.logfbank(audio_1, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            ssc_feat_a1 = psf.logfbank(audio_1, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            full_feat_1 = np.concatenate([mfcc_a1, logfbank_a1, ssc_feat_a1], axis=1)
            # generate the audio features for speaker 2
            mfcc_a2 = psf.mfcc(audio_2, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
            logfbank_a2 = psf.logfbank(audio_2, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            ssc_feat_a2 = psf.logfbank(audio_2, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            full_feat_2 = np.concatenate([mfcc_a2, logfbank_a2, ssc_feat_a2], axis=1)
            # generate the audio features for speaker 0 other
            mfcc_a0_other = psf.mfcc(audio_0_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
            logfbank_a0_other = psf.logfbank(audio_0_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            ssc_feat_a0_other = psf.logfbank(audio_0_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            full_feat_0_other = np.concatenate([mfcc_a0_other, logfbank_a0_other, ssc_feat_a0_other], axis=1)
            # generate the audio features for speaker 1 other
            mfcc_a1_other = psf.mfcc(audio_1_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
            logfbank_a1_other = psf.logfbank(audio_1_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            ssc_feat_a1_other = psf.logfbank(audio_1_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            full_feat_1_other = np.concatenate([mfcc_a1_other, logfbank_a1_other, ssc_feat_a1_other], axis=1)
            # generate the audio features for speaker 2 other
            mfcc_a2_other = psf.mfcc(audio_2_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
            logfbank_a2_other = psf.logfbank(audio_2_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            ssc_feat_a2_other = psf.logfbank(audio_2_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
            full_feat_2_other = np.concatenate([mfcc_a2_other, logfbank_a2_other, ssc_feat_a2_other], axis=1)
            words = json.load(open(word_output_path, "r"))
            # get transcript for each
            text_0 = words["0"]
            text_1 = words["1"]
            text_2 = words["2"]
            text_0_other = words["0_other"]
            text_1_other = words["1_other"]
            text_2_other = words["2_other"]
            ts_target = np.arange(0, full_feat_1.shape[0]) / target_fps
            # get word vector
            word_structure_feature_0 = generate_word_structure_values(ts_target, text_0)
            word_structure_feature_1 = generate_word_structure_values(ts_target, text_1)
            word_structure_feature_2 = generate_word_structure_values(ts_target, text_2)
            word_structure_feature_0_other = generate_word_structure_values(ts_target, text_0_other)
            word_structure_feature_1_other = generate_word_structure_values(ts_target, text_1_other)
            word_structure_feature_2_other = generate_word_structure_values(ts_target, text_2_other)
            
            # get the sentence vector
            interval_of_sentences_0, ___ = parse_for_sentence_intervals(text_0, 0.56)
            interval_of_sentences_1, ___ = parse_for_sentence_intervals(text_1, 0.56)
            interval_of_sentences_2, ___ = parse_for_sentence_intervals(text_2, 0.56)
            interval_of_sentences_0_other, ___ = parse_for_sentence_intervals(text_0_other, 0.56)
            interval_of_sentences_1_other, ___ = parse_for_sentence_intervals(text_1_other, 0.56)
            interval_of_sentences_2_other, ___ = parse_for_sentence_intervals(text_2_other, 0.56)
            
            sentence_structure_feature_0 = generate_sentence_structure_values_real(ts_target, interval_of_sentences_0)
            sentence_structure_feature_1 = generate_sentence_structure_values_real(ts_target, interval_of_sentences_1)
            sentence_structure_feature_2 = generate_sentence_structure_values_real(ts_target, interval_of_sentences_2)
            sentence_structure_feature_0_other = generate_sentence_structure_values_real(ts_target, interval_of_sentences_0_other)
            sentence_structure_feature_1_other = generate_sentence_structure_values_real(ts_target, interval_of_sentences_1_other)
            sentence_structure_feature_2_other = generate_sentence_structure_values_real(ts_target, interval_of_sentences_2_other)
            
            # get the output feature
            output_feature_just_0 = np.concatenate([full_feat_0, word_structure_feature_0, sentence_structure_feature_0], axis=1)
            output_feature_just_1 = np.concatenate([full_feat_1, word_structure_feature_1, sentence_structure_feature_1], axis=1)
            output_feature_just_2 = np.concatenate([full_feat_2, word_structure_feature_2, sentence_structure_feature_2], axis=1)
            output_feature_just_0_other = np.concatenate([full_feat_0_other, word_structure_feature_0_other, sentence_structure_feature_0_other], axis=1)
            output_feature_just_1_other = np.concatenate([full_feat_1_other, word_structure_feature_1_other, sentence_structure_feature_1_other], axis=1)
            output_feature_just_2_other = np.concatenate([full_feat_2_other, word_structure_feature_2_other, sentence_structure_feature_2_other], axis=1)
            
            output_feature_0 = np.concatenate([output_feature_just_0, output_feature_just_0_other], axis=1)
            output_feature_1 = np.concatenate([output_feature_just_1, output_feature_just_1_other], axis=1)
            output_feature_2 = np.concatenate([output_feature_just_2, output_feature_just_2_other], axis=1)
            np.save(processed_input_vector_0_path, output_feature_0)
            np.save(processed_input_vector_1_path, output_feature_1)
            np.save(processed_input_vector_2_path, output_feature_2)
        # also output the file into the temp folder for JALI to generate the lip motion
        if not os.path.exists(script_output_path_0) :
            script = json.load(open(word_output_path, "r"))
            script_0 = script["0"]
            # get the text of speaker 0
            script_out_0 = ""
            for i in range(0, len(script_0)):
                script_out_0 += " " + script_0[i]["text"]
            
            script_1 = script["1"]
            # get the text of speaker 1
            script_out_1 = ""
            for i in range(0, len(script_1)):
                script_out_1 += " " + script_1[i]["text"]
            
            script_2 = script["2"]
            # get the text of speaker 2
            script_out_2 = ""
            for i in range(0, len(script_2)):
                script_out_2 += " " + script_2[i]["text"]
                
            # store the script for JALI
            with open(script_output_path_0, "w") as file:
                file.write(script_out_0)
            with open(script_output_path_1, "w") as file:
                file.write(script_out_1)
            with open(script_output_path_2, "w") as file:
                file.write(script_out_2)
            # generate the tagged version as well.
            shutil.copy2(script_output_path_0, script_tagged_output_path_0)
            shutil.copy2(script_output_path_1, script_tagged_output_path_1)
            shutil.copy2(script_output_path_2, script_tagged_output_path_2)
        if not os.path.exists(tagged_dialog_output_path_0):
            with open(word_output_path, "r") as f:
                transcribe_json = json.load(f)
            speaker_0 = transcribe_json["0"]
            speaker_1 = transcribe_json["1"]
            speaker_2 = transcribe_json["2"]
            speaker_0_other = transcribe_json["0_other"]
            speaker_1_other = transcribe_json["1_other"]
            speaker_2_other = transcribe_json["2_other"]
            output_string_0 = merge_transcript(speaker_0, speaker_0_other)
            output_string_1 = merge_transcript(speaker_1, speaker_1_other)
            output_string_2 = merge_transcript(speaker_2, speaker_2_other)
            
            with open(tagged_dialog_output_path_0, "w") as f:
                f.write(output_string_0)
            with open(tagged_dialog_output_path_1, "w") as f:
                f.write(output_string_1)
            with open(tagged_dialog_output_path_2, "w") as f:
                f.write(output_string_2)
        torch.set_default_tensor_type(torch.DoubleTensor)
        if speaker == 0:
            X = torch.from_numpy(np.expand_dims(np.load(processed_input_vector_0_path), axis=0)).to(self.device)
        elif speaker == 1:
            X = torch.from_numpy(np.expand_dims(np.load(processed_input_vector_1_path), axis=0)).to(self.device)
        elif speaker == 2:
            X = torch.from_numpy(np.expand_dims(np.load(processed_input_vector_2_path), axis=0)).to(self.device)
        out = self.model(X)[0].cpu().detach().numpy()
        out = softmax(out, axis=1)
        return out, X
                    
def generate_word_alignment(praat_path):
    
    phone_prosodic_list = []
    phone_list = []
    phone_intervals = []
    word_list = []
    word_intervals = []
    stats = []
    try:
        f = open(praat_path)
    except:
        print("check if file exist at \n {}".format(praat_path))
        return [], [], [], []
    garb = f.readline()
    arr = f.readlines()
    for i in range(0, len(arr)):
        content = arr[i].split("\t")
        start = float(content[0])
        end = float(content[1])
        phone = content[21]
        word = content[26]
        phone_list.append(phone)
        word_list.append(word)
        phone_intervals.append([start, end])
    #     merged_word_list = []
    merged_word_intervals = []

    prev_word = word_list[0]
    merged_word_intervals.append([phone_intervals[0][0]])

    prev_k = 1
    for i in range(1, len(word_list)):
        if word_list[i] != prev_word:
            merged_word_intervals[-1].append(phone_intervals[i - 1][1])
            for j in range(0, prev_k - 1):
                merged_word_intervals.append(merged_word_intervals[-1])
            prev_word = word_list[i]
            merged_word_intervals.append([phone_intervals[i][0]])
            prev_k = 0
        if i == len(word_list) - 1:
            merged_word_intervals[-1].append(phone_intervals[i - 1][1])
            for j in range(0, prev_k):
                merged_word_intervals.append(merged_word_intervals[-1])
        prev_k = prev_k + 1
        
        
    out_word_list = []
    out_word_intervals = []
    start = 0
    for i in range(1, len(word_list)):
        if word_list[i - 1] != word_list[i]:
            out_word_list.append(word_list[i - 1])
            out_word_intervals.append(merged_word_intervals[i - 1])
            word_to_phone_temp = []
            for j in range(start, i):
                word_to_phone_temp.append(j)
            start = i
        if i == len(word_list) - 1:

            out_word_list.append(word_list[i])
            out_word_intervals.append(merged_word_intervals[i])
            word_to_phone_temp = []
            for j in range(start, i + 1):
                word_to_phone_temp.append(j)
            start = i
    return phone_list, phone_intervals, out_word_list, out_word_intervals
def generate_transcript_whisper_style(word_list, word_intervals):
    output = []
    for i in range(0, len(word_list)):
        if word_list[i] == ".":
            continue
        elif word_list[i][0] == "_":
            output.append({"text":word_list[i][1:], "start":word_intervals[i][0], "end":word_intervals[i][1], "confidence":0.5})
        else:
            output.append({"text":word_list[i][:], "start":word_intervals[i][0], "end":word_intervals[i][1], "confidence":0.5})
    return output
def combine_whisper_transcripts(transcript_1, transcript_2):
    transcript_combined = []
    pointer_1 = 0
    pointer_2 = 0
    while pointer_1 < len(transcript_1) and pointer_2 < len(transcript_2):
        if transcript_1[pointer_1]["start"] < transcript_2[pointer_2]["start"]:
            transcript_combined.append(transcript_1[pointer_1])
            pointer_1 = pointer_1 + 1
        else:
            transcript_combined.append(transcript_2[pointer_2])
            pointer_2 = pointer_2 + 1
    while pointer_1 < len(transcript_1):
        transcript_combined.append(transcript_1[pointer_1])
        pointer_1 = pointer_1 + 1
    while pointer_2 < len(transcript_2):
        transcript_combined.append(transcript_2[pointer_2])
        pointer_2 = pointer_2 + 1
    return transcript_combined
                       
            
            
            

            
            
            
        
                    
                    