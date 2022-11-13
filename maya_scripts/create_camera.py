# create camera in the character's perspective
import pymel.core as pm
import maya.cmds as cmds
import math
import maya.OpenMaya as om
import maya.OpenMayaUI as omUI

import json

def obtain_items_of_interest(manual_select):
    try:
        cmds.delete("camera1")
    except:
        pass
    out_position_list = []
    out_selected_item_names_list = []
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
    for i in range(0, 3):
        head_direction[i] = head_direction[i] / magnitude
        head_position[i] = head_position[i] + 20 * head_direction[i]
    
    # create a camera
    cameraName = cmds.camera()
    cameraShape = cameraName[1]
    print(head_position)
    # set camera location and look at point
    cmds.viewPlace(cameraShape, eye=head_position, la=look_at_point_position)
    # look through the camera
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
    # output the position of items
    output_json_item_position = {}
    for i in range(0, len(item_names)):
        output_json_item_position[item_names[i]] = item_positions[i]
    # output the category of items
    output_json_item_type = {}
    for i in range(0, len(item_names)):
        try:
            output_json_item_type[item_names[i]] = cmds.getAttr(item_names[i]+".Object_type")
        except:
            output_json_item_type[item_names[i]] = 0
    # output the interestingness of the item:
    output_json_item_interesting_ness = {}
    for i in range(0, len(item_names)):
        try:
            output_json_item_interesting_ness[item_names[i]] = cmds.getAttr(item_names[i]+".Interestingness")
        except:
            output_json_item_interesting_ness[item_names[i]] = 0.0
    # output the position of the head
    output_json_self = {
        "pos": pm.xform("eyeStare_zero",q=True,t=True,ws=True),
        "dir": head_direction
    }
    output_json = {
        "self_pos":output_json_self, 
        "object_pos":output_json_item_position,
        "object_type": output_json_item_type,
        "object_interestingness": output_json_item_interesting_ness
    }
    with open("C:/Users/evansamaa/Desktop/Gaze_project/data/look_at_points/simplest_scene.json", "w") as f:
        json.dump(output_json, f) 

    return out_selected_item_names_list, out_position_list
        
        
        
item_names, item_positions = obtain_items_of_interest(False)
         