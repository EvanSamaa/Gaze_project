from prototypes.VirtualClasses import Base_Static_Saliency_List
from prototypes.InputDataStructures import Dietic_Conversation_Gaze_Scene_Info
import numpy as np
from Speech_Data_util import Sentence_word_phone_parser
from prototypes.MVP.MVP_static_saliency_list import ObjectBasedFixSaliency
from matplotlib import pyplot as plt
class HabituationBasedPlanner():
    def __init__(self, saliency_list: ObjectBasedFixSaliency, audio: np.array, script: Sentence_word_phone_parser, scene_info: Dietic_Conversation_Gaze_Scene_Info):
        self.saliency_list = saliency_list
        self.scene_info = scene_info
        self.script = script
        self.audio = audio

        self.t = 0
        self.current_look_at_target = 0
        self.current_look_at_position = self.scene_info.speaker_face_direction
        ############################################################
        ############## habituation related parameters ##############
        ############################################################
        self.current_habituation = self.saliency_list.evaluate(0)
        # the interest in any object would wain after 3 seconds
        self.mhab = np.ones((self.saliency_list._number_of_objects, )) * 0.3
        # the interest would be return to 1 after 3 seconds
        self.mrest = np.ones((self.saliency_list._number_of_objects,)) * 0.3

        ############################################################
        ################### SCENE related things ###################
        ############################################################
        # get listener target position
        listener_direction_l = np.zeros((3, ))
        # setup habituation factor
        for i in range(0, len(self.scene_info.scene_object_id)):
            obj_id = self.scene_info.scene_object_id[i]
            obj_type = self.scene_info.object_type[obj_id]

            if (obj_type == 1):
                self.mhab[i] /= float(self.saliency_list._number_of_objects)
                self.mrest[i] *= float(self.saliency_list._number_of_objects)
                pass
            else:
                self.mhab[i] /= self.saliency_list.evaluate_all()[0, i]
                self.mrest[i] *= self.saliency_list.evaluate_all()[0, i]
                pass


    def compute_habituation(self, dt):
        """
        get the habituation for all targets based on time passed and gaze_target
        this is implemented as per "Realistic and Interactive Robot Gaze, Pan, 2020"
        :param dt: time passed
        :return:
        """
        tao = np.zeros((self.saliency_list.map.shape[1],))
        tao_not = np.ones((self.saliency_list.map.shape[1],))
        tao[self.current_look_at_target] = 1
        tao_not[self.current_look_at_target] = 0
        new_habituation = self.current_habituation + ( -self.mhab * tao + tao_not * self.mrest) * dt
        self.t += dt
        self.current_look_at_target = np.argmax(new_habituation * self.saliency_list.evaluate(self.t))
        self.current_look_at_position = self.scene_info.object_pos[self.scene_info.scene_object_id[self.current_look_at_target]]
        self.current_habituation = new_habituation
        return self.current_look_at_target
    def compute(self):
        output = []
        prev_time = 0
        hab = []
        for i in range(0, len(self.script.word_list)):
            dt = self.script.word_intervals[i][0] - prev_time
            output.append(self.compute_habituation(dt))
            hab.append(np.expand_dims(self.current_habituation * self.saliency_list.evaluate(self.t), axis=0))
            prev_time = self.script.word_intervals[i][0]
        hab = np.concatenate(hab, axis = 0)
        print(output)
        plt.plot(hab)
        plt.show()






if __name__ == "__main__":
    planner = HabituationBasedPlanner()