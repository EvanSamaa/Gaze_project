from Speech_Data_util import Check_Pauses, Sentence_word_phone_parser, XSampa_phonemes_dicts
import numpy as np
class ComputeAversionProbability:
    def __init__(self, script, audio):
        # set up audio and script
        self.script: Sentence_word_phone_parser = script
        self.audio = audio

        # set up helper classes
        self.isPause = Check_Pauses()
        self.phone_dict = XSampa_phonemes_dicts()

        # set up hyper-parameters
        self.eye_contact_comfort_level = 5  # how much time the character is comfortable with holding eye contact
        self.gaze_away_comfort_level = 5  # how often the character is comfortable with looking else where until looking at the charcter

    def compute(self):
        # set state variables
        time_since_prev_aversion = 5
        time_since_gaze_averted = 0
        target = -1

        avert_probability_value = []
        avert_probability_time = []
        # compute word_intervals
        word_intervals = []
        j = 0
        for i in range(0, len(self.script.word_list)):
            while  j < len(self.script.phone_list) and self.script.word_list[i] == self.script.phone_to_word[j]:
                word_intervals.append(self.script.word_intervals[i])
                j = j+1

        for i in range(1, len(self.script.word_list)):
            word = self.script.word_list[i]

            # if there is any pauses/filled pauses, we assume that they are thinking. and we
            # make them look at something else
            # TODO: More elaborate triggers shall be designed
            if ((self.isPause(word) == "FP" or
                 (self.isPause(word) == "SP" and word_intervals[i][1] - word_intervals[i][0] >= 0.2))
                    and time_since_prev_aversion >= self.eye_contact_comfort_level):
                time_since_prev_aversion = 0
                time_since_gaze_averted = self.script.word_intervals[i][1] - self.script.word_intervals[i][0]
                # will start looking elsewhere
                avert_probability_value.append(1)
                avert_probability_time.append(self.script.word_intervals[i][0])


            # if the target is away, look back when it has looked away for too long
            elif len(avert_probability_value) > 0 and avert_probability_value[-1] == 1:
                if self.phone_dict.strip(
                        self.script.phone_list[i]) in self.phone_dict.vowels and time_since_gaze_averted >= self.gaze_away_comfort_level:
                    avert_probability_value.append(0)
                    avert_probability_time.append(self.script.word_intervals[i][0])
                    time_since_gaze_averted = 0
                else:
                    time_since_gaze_averted += self.script.word_intervals[i][1] - self.script.word_intervals[i][0]
            else:
                avert_probability_value.append(0)
                avert_probability_time.append(self.script.word_intervals[i][0])
        return avert_probability_time, avert_probability_value


if __name__ == "__main__":
    k = 2