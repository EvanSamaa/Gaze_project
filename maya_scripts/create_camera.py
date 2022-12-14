# create camera in the character's perspective
import pymel.core as pm
import maya.cmds as cmds
import math
import maya.OpenMaya as om
import maya.OpenMayaUI as omUI

import json

def obtain_items_of_interest(manual_select, look_through=True):
    try:
        cmds.delete("camera1")
    except:
        pass
    out_position_list = []
    out_selected_item_names_list = []
    cmds.setAttr("eyeStare_world.eyeStare_world_translateX", 0)
    cmds.setAttr("eyeStare_world.eyeStare_world_translateY", 0)
    cmds.setAttr("eyeStare_world.eyeStare_world_translateZ", 100)
    # get position of the character and the look at point
    head_position = pm.xform("eyeStare_zero",q=True,t=True,ws=True)
    look_at_point_position = pm.xform("eyeStare_world",q=True,t=True,ws=True)
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
    # create a camera
    cameraName = cmds.camera()
    cameraShape = cameraName[1]
    # set camera location and look at point
    cmds.viewPlace(cameraShape, eye=for_camera_head_position , la=look_at_point_position)
    # look through the camera
    if look_through:
        current_panel = cmds.getPanel(wf = True)
        cmds.setAttr(cameraShape+".focalLength", 3);
        cmds.lookThru(current_panel, cameraShape)
    
    # select items in the camera
    if not manual_select:
        view = omUI.M3dView.active3dView()
        items = om.MGlobal.selectFromScreen( 0, 0, view.portWidth(), view.portHeight(), om.MGlobal.kReplaceList)
        selected_item_names_list = cmds.ls(selection=True)
    else:
        selected_item_names_list= cmds.ls(selection=True)
    filtered = cmds.filterExpand( ex=False, sm=(12))
    cmds.select(filtered)
    for item in filtered:
        if str(item[0]) == "p":
            item_position = pm.xform(item,q=True,t=True,ws=True)
            out_position_list.append(item_position)
            out_selected_item_names_list.append(item)
    cmds.select(out_selected_item_names_list)
    # output the world space position of items
    output_json_item_position = {}
    for i in range(0, len(out_selected_item_names_list)):
        output_json_item_position[out_selected_item_names_list[i]] = out_position_list[i]
    # output the category of items
    output_json_item_type = {}
    for i in range(0, len(out_selected_item_names_list)):
        try:
            output_json_item_type[out_selected_item_names_list[i]] = cmds.getAttr(item_names[i]+".ObjectType")
        except:
            output_json_item_type[out_selected_item_names_list[i]] = 0
    # output the interestingness of the item:
    output_json_item_interesting_ness = {}
    for i in range(0, len(out_selected_item_names_list)):
        try:
            output_json_item_interesting_ness[out_selected_item_names_list[i]] = cmds.getAttr(item_names[i]+".Interest")
        except:
            output_json_item_interesting_ness[out_selected_item_names_list[i]] = 0.00000001
    # output the position of the head
    
    calibration_dir_local = [0, 0, 100]
    output_json_self = {
        "pos": pm.xform("eyeStare_zero",q=True,t=True,ws=True),
        "calibration_dir_local": calibration_dir_local,
        "calibration_dir_world": look_at_point_position        
    }       
    output_json = {
        "self_pos":output_json_self, 
        "object_pos":output_json_item_position,
        "object_type": output_json_item_type,
        "object_interestingness": output_json_item_interesting_ness
    }
    with open("C:/Users/evansamaa/Desktop/Gaze_project/data/look_at_points/simplest_scene.json", "w") as f:
    # with open("C:/Users/evan1/Documents/Gaze_project/data/look_at_points/simplest_scene.json", "w") as f:
        json.dump(output_json, f) 
    if look_through:
        current_panel = cmds.getPanel(wf = True)
        cmds.lookThru(current_panel, "persp")
    return out_selected_item_names_list, out_position_list
        
        
        
item_names, item_positions = obtain_items_of_interest(False)
print("at the end")     