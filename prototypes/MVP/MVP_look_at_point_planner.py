import random

from prototypes.VirtualClasses import Base_Static_Saliency_List
from prototypes.InputDataStructures import Dietic_Conversation_Gaze_Scene_Info
import numpy as np
from Speech_Data_util import Sentence_word_phone_parser
from prototypes.MVP.MVP_static_saliency_list import ObjectBasedFixSaliency
from matplotlib import pyplot as plt
class HabituationBasedPlanner():
    def __init__(self, saliency_list: ObjectBasedFixSaliency, audio: np.array, script: Sentence_word_phone_parser, scene_info: Dietic_Conversation_Gaze_Scene_Info, engage_threshold = 0.8):
        self.saliency_list = saliency_list
        self.scene_info = scene_info
        self.script = script
        self.audio = audio
        self.engage_threshold = engage_threshold

        self.t = 0
        self.current_look_at_target = 0
        self.current_look_at_position = self.scene_info.speaker_face_direction_local
        ############################################################
        ############## habituation related parameters ##############
        ############################################################
        self.current_habituation = self.saliency_list.evaluate(0)
        # the interest in any object would wain after 3 seconds
        self.mhab = np.ones((self.saliency_list._number_of_objects, )) * 0.2
        # the interest would be return to 1 after 1 seconds
        self.mrest = np.ones((self.saliency_list._number_of_objects,)) * 1.0
        ############################################################
        ################### SCENE related things ###################
        ############################################################
        # get listener target position
        self.conversation_partner_mask = np.zeros((self.saliency_list._number_of_objects, ))


        # setup habituation factor
        for i in range(0, len(self.scene_info.scene_object_id)):
            obj_type = self.scene_info.object_type[i]
            if (obj_type == 5):
                self.conversation_partner_mask[obj_type] = 1
                # self.mhab[i] /= float(self.saliency_list._number_of_objects)
                # self.mhab[i] = self.mhab[i]
                # self.mrest[i] *= float(self.saliency_list._number_of_objects)
                pass
            else:
                # self.mhab[i] /= self.saliency_list.evaluate_all()[0, i]
                # self.mrest[i] *= self.saliency_list.evaluate_all()[0, i]
                pass


    def compute_habituation(self, dt):
        """
        get the habituation for all targets based on time passed and gaze_target
        this is implemented as per "Realistic and Interactive Robot Gaze, Pan, 2020"
        :param dt: time passed
        :return:
        """
        # let's make the conversation partner's habituation decay, but the other objects behave in different ways
        tao = np.zeros((self.saliency_list.map.shape[1],))
        tao_not = np.ones((self.saliency_list.map.shape[1],))
        tao[self.current_look_at_target] = 1
        tao_not[self.current_look_at_target] = 0
        new_habituation = self.current_habituation + ( -self.mhab * tao + tao_not * self.mrest) * dt
        new_curiosity_score = new_habituation * self.saliency_list.evaluate(self.t)
        self.t += dt
        if np.max(new_curiosity_score) >= self.engage_threshold:
            self.current_look_at_target = np.argmax(new_curiosity_score)
            self.current_look_at_position = self.scene_info.object_pos[self.current_look_at_target]
            self.current_habituation = new_habituation
        else:
            self.current_look_at_target = np.argmax(self.scene_info.object_type)
            self.current_look_at_position = self.scene_info.object_pos[self.current_look_at_target]
            self.current_habituation = new_habituation
        return self.current_look_at_target
    def compute(self):
        output_targets = []
        output_times = []
        prev_time = 0
        hab = []
        for i in range(0, len(self.script.word_list)):
            dt = self.script.word_intervals[i][0] - prev_time
            target = self.compute_habituation(dt)
            output_targets.append(target)
            hab.append(np.expand_dims(self.current_habituation * self.saliency_list.evaluate(self.t), axis=0))
            output_times.append(self.t)
            prev_time = self.script.word_intervals[i][0]

        hab = np.concatenate(hab, axis = 0)
        plt.plot(output_times, hab)
        plt.show()
        return output_times, output_targets
class PartnerHabituationPlanner():
    def __init__(self, saliency_list: ObjectBasedFixSaliency, audio: np.array, script: Sentence_word_phone_parser, scene_info: Dietic_Conversation_Gaze_Scene_Info, engage_threshold = 0.8):
        self.saliency_list = saliency_list
        self.scene_info = scene_info
        self.script = script
        self.audio = audio
        self.engage_threshold = engage_threshold

        # state variables
        self.t = 0
        self.current_look_at_target = 0
        self.current_look_at_position = self.scene_info.speaker_face_direction_local

        ############################################################
        ################### SCENE related things ###################
        ############################################################

        # get listener target position
        self.conversation_partner_mask = np.zeros((self.saliency_list._number_of_objects, ))
        self.obj_mask = np.ones((self.saliency_list._number_of_objects,))
        self.conversation_partner_id = self.scene_info.object_type.argmax()
        self.conversation_partner_mask[self.conversation_partner_id] = 1
        self.obj_mask[self.conversation_partner_id] = 0

        ############################################################
        ############## habituation related parameters ##############
        ############################################################
        self.current_habituation = 1
        self.current_object_habituation = 0.5 * np.ones(self.scene_info.object_interest.shape) * self.obj_mask
        # the interest in any object would wain after 3 seconds
        self.mhab = 0.10
        # the interest would be return to 1 after 1 seconds
        self.mrest = 1.0

        # interest on an object would wane to zero after 1 second
        self.obj_mhab = np.ones(self.scene_info.object_interest.shape) * 1.5
        self.obj_mhab *= self.obj_mask
        # the interest will be slowly gained over time
        self.obj_mrest = np.ones(self.scene_info.object_interest.shape) * 0.01
        self.obj_mrest *= self.obj_mask


    def compute_habituation(self, dt):
        """
        get the habituation for all targets based on time passed and gaze_target
        this is implemented as per "Realistic and Interactive Robot Gaze, Pan, 2020"
        :param dt: time passed
        :return:
        """
        # let's make the conversation partner's habituation decay, but the other objects behave in different ways
        if self.current_look_at_target == self.conversation_partner_id:
            tao_speaker = 1
            tao_not_speaker = 0
            tao_item = np.zeros(self.obj_mask.shape)
            tao_not_item = self.obj_mask
        else:
            tao_speaker = 0
            tao_not_speaker = 1
            tao_item = np.zeros(self.obj_mask.shape)
            tao_item[self.current_look_at_target] = 1
            tao_not_item = self.obj_mask
            tao_not_item[self.current_look_at_target] = 0

        self.current_habituation= self.current_habituation + ( -self.mhab * tao_speaker + tao_not_speaker * self.mrest) * dt
        self.current_object_habituation = self.current_object_habituation = (-self.obj_mhab * tao_item + tao_not_item * self.obj_mrest) * dt

    def compute(self):
        output_targets = []
        output_times = []
        prev_time = 0
        hab = []
        for i in range(0, len(self.script.word_list)):
            dt = self.script.word_intervals[i][0] - prev_time
            self.t += dt
            self.compute_habituation(dt)
            if self.current_habituation >= 1:
                self.current_look_at_target = self.conversation_partner_id
                output_targets.append(self.current_look_at_target)
            else:
                target = np.argmax(self.saliency_list.evaluate(self.t) * self.current_object_habituation * self.obj_mask)
                output_targets.append(target)
                self.current_look_at_target = target
            output_times.append(self.t)
            prev_time = self.script.word_intervals[i][0]
        # plt.plot(output_times, hab)
        # plt.show()
        return output_times, output_targets
class RandomPlanner():
    def __init__(self, saliency_list: ObjectBasedFixSaliency, audio: np.array, script: Sentence_word_phone_parser,
                 scene_info: Dietic_Conversation_Gaze_Scene_Info, engage_threshold=0.8):
        self.saliency_list = saliency_list
        self.scene_info = scene_info
        self.script = script
        self.audio = audio
        self.engage_threshold = engage_threshold

        self.t = 0
        self.current_look_at_target = 0
        self.current_look_at_position = self.scene_info.speaker_face_direction_local
        ############################################################
        ############## habituation related parameters ##############
        ############################################################
        self.current_habituation = self.saliency_list.evaluate(0)
        # the interest in any object would wain after 3 seconds
        self.mhab = np.ones((self.saliency_list._number_of_objects,)) * 0.3
        # the interest would be return to 1 after 3 seconds
        self.mrest = np.ones((self.saliency_list._number_of_objects,)) * 1.0
        ############################################################
        ################### SCENE related things ###################
        ############################################################
        # get listener target position
        self.conversation_partner_mask = np.zeros((self.saliency_list._number_of_objects,))
        self.conversation_partner_index = self.scene_info.object_type.argmax()
        self.conversation_partner_mask[self.conversation_partner_index] = 1
    def compute(self):
        output_targets = []
        output_times = []
        prev_time = 0
        for i in range(0, len(self.script.sentence_intervals)):
            # get the starting time (first word that is not a silence)
            starting_word_interval_id = self.script.sentence_to_word[i][0]


            t = self.script.sentence_intervals[i][0]
            if random.random() >= self.engage_threshold:
                obj_to_look = random.choice(range(0, len(self.scene_info.scene_object_id)))
                while(obj_to_look == self.conversation_partner_index):
                    obj_to_look = random.choice(range(0, len(self.scene_info.scene_object_id)))
                output_targets.append(obj_to_look)
                output_times.append(t)
            else:
                output_targets.append(self.conversation_partner_index)
                output_times.append(t)
        print(output_targets)
        return output_times, output_targets

if __name__ == "__main__":
    # planner = HabituationBasedPlanner()
    pass