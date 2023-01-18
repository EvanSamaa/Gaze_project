# create camera in the character's perspective
import pymel.core as pm
import maya.cmds as cmds
import math
import maya.OpenMaya as om
import maya.OpenMayaUI as omUI
import json

def obtain_items_of_interest_for_three_speaker_scene(manual_select, look_through=True):
    character_neck_node = ["jNeck_ctl", "Mike|pasted__JALI_GRP|pasted__jNeck_GRP|pasted__jNeck_parent|pasted__jNeck_ctl", "Jessie|pasted__JALI_GRP|pasted__jNeck_GRP|pasted__jNeck_parent|pasted__jNeck_ctl"]
    character_gaze_node = ["eyeStare_world", 
    "Jessie|pasted__JALI_GRP|pasted__jRig_GRP|pasted__masterGRP|pasted__lookMaster_GRP|pasted__lookMaster_outGRP|pasted__eyeStare_World_GRP|pasted__zero_eyeStare_World|pasted__eyeStare_world",
    "Mike|pasted__JALI_GRP|pasted__jRig_GRP|pasted__masterGRP|pasted__lookMaster_GRP|pasted__lookMaster_outGRP|pasted__eyeStare_World_GRP|pasted__zero_eyeStare_World|pasted__eyeStare_world"]
    character_gaze_stare_zero_node = ["eyeStare_zero", 
    "Jessie|pasted__JALI_GRP|pasted__jRig_GRP|pasted__masterGRP|pasted__lookMaster_GRP|pasted__lookMaster_outGRP|pasted__eyeStare_zero",
    "Mike|pasted__JALI_GRP|pasted__jRig_GRP|pasted__masterGRP|pasted__lookMaster_GRP|pasted__lookMaster_outGRP|pasted__eyeStare_zero"]
    
    out_object_names = []
    out_positions = {}
    out_interesting_ness = {}
    out_calibration_local = [0, 0, 100]
    out_calibration_global = {}
    out_object_type = {}
    speaker_indexes = []
    speaker_facing_dire = {}
    
    view = omUI.M3dView.active3dView()
    items = om.MGlobal.selectFromScreen( 0, 0, view.portWidth(), view.portHeight(), om.MGlobal.kReplaceList)
    selected_item_names_list = cmds.ls(selection=True)
    filtered = cmds.filterExpand( ex=False, sm=(12))
    cmds.select(selected_item_names_list)
    out_position_list = []
    out_selected_item_names_list = []
    for item in filtered:
        if str(item[0]) == "p" and str(item[0:6]) != "pasted":
            print(item)
            item_position = pm.xform(item,q=True,t=True,ws=True)
            out_position_list.append(item_position)
            out_selected_item_names_list.append(item)
    cmds.select(out_selected_item_names_list)
    # output the world space position of items
    output_json_item_position = {}
    for i in range(0, len(out_selected_item_names_list)):
        out_object_names.append(out_selected_item_names_list[i])
        out_positions[i + len(character_gaze_stare_zero_node)] = out_position_list[i]
    # output the category of items
    for i in range(0, len(out_selected_item_names_list)):
        try:
            out_object_type[i + len(character_gaze_stare_zero_node)] = cmds.getAttr(item_names[i]+".ObjectType")
        except:
            out_object_type[i + len(character_gaze_stare_zero_node)] = 0
    for i in range(0, len(character_gaze_stare_zero_node)):
        out_object_type[i] = 5
    # output the interestingness of the item:
    for i in range(0, len(out_selected_item_names_list)):
        try:
            out_interesting_ness[i + len(character_gaze_stare_zero_node)] = cmds.getAttr(item_names[i]+".Interest")
        except:
            out_interesting_ness[i + len(character_gaze_stare_zero_node)] = 0.00000001
    for i in range(0, len(character_gaze_stare_zero_node)):
        out_interesting_ness[i] = 0.8
    for iii in range(0, len(character_neck_node)):
        try:
            cmds.delete("camera1")
        except:
            pass
        out_position_list = []
        out_selected_item_names_list = []
        cmds.setAttr("{}.eyeStare_world_translateX".format(character_gaze_node[iii]), 0)
        cmds.setAttr("{}.eyeStare_world_translateY".format(character_gaze_node[iii]), 0)
        cmds.setAttr("{}.eyeStare_world_translateZ".format(character_gaze_node[iii]), 100)
        # get position of the character and the look at point
        cmds.select(character_gaze_stare_zero_node[iii])
        head_position = pm.xform(character_gaze_stare_zero_node[iii],q=True,t=True,ws=True)
        look_at_point_position = pm.xform(character_gaze_node[iii],q=True,t=True,ws=True)
        head_direction =  [0, 0, 0]
        magnitude = 0
        for i in range(0, 3):
            head_direction[i] = look_at_point_position[i] - head_position[i]
            magnitude = magnitude + head_direction[i] * head_direction[i]
        # normalize it to magnitude of 1
        magnitude = math.sqrt(magnitude)
        for_camera_head_position = [0, 0, 0]
        for i in range(0, 3):
            head_direction[i] = head_direction[i] / magnitude
            for_camera_head_position [i] = head_position[i] + 20 * head_direction[i]
        # output the position of the head
        
        calibration_dir_local = [0, 0, 100]
        out_calibration_global[iii] = look_at_point_position
        out_positions[iii] = pm.xform(character_gaze_stare_zero_node[iii],q=True,t=True,ws=True)
        speaker_indexes.append(iii)
    output_json = {
        "object_pos":out_positions,
        "object_type": out_object_type,
        "object_interestingness": out_interesting_ness,
        "speaker_indexes": speaker_indexes,
        "calibration_global": out_calibration_global,
        "calibration_local": out_calibration_local,
        "object_names": out_object_names
    }
    with open("C:/Users/evansamaa/Desktop/Gaze_project/data/look_at_points/three_party.json", "w") as f:
    # with open("C:/Users/evan1/Documents/Gaze_project/data/look_at_points/simplest_scene.json", "w") as f:
        json.dump(output_json, f) 
    if look_through:
        current_panel = cmds.getPanel(wf = True)
        cmds.lookThru(current_panel, "persp")
    return
            
            
        
obtain_items_of_interest_for_three_speaker_scene([], False)
print("at the end")     