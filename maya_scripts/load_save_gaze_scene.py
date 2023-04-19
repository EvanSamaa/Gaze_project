import maya.cmds as cmds
import json
import math
# Define the load_scene function
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
    
    # select items in the camera
    if not manual_select:
        view = omUI.M3dView.active3dView()
        items = om.MGlobal.selectFromScreen( 0, 0, view.portWidth(), view.portHeight(), om.MGlobal.kReplaceList)
        selected_item_names_list = cmds.ls(selection=True)
    else:
        selected_item_names_list= cmds.ls(selection=True)
    print(selected_item_names_list)
    for item in selected_item_names_list:
        item_position = pm.xform(item,q=True,t=True,ws=True)
        print(item_position)
        out_position_list.append(item_position)
        out_selected_item_names_list.append(item)
    # output the world space position of items
    output_json_item_position = {}
    for i in range(0, len(out_selected_item_names_list)):
        output_json_item_position[out_selected_item_names_list[i]] = out_position_list[i]
    # output the category of items
    output_json_item_type = {}
    for i in range(0, len(out_selected_item_names_list)):
        try:
            output_json_item_type[out_selected_item_names_list[i]] = cmds.getAttr(out_selected_item_names_list[i]+".object_type")
        except:
            output_json_item_type[out_selected_item_names_list[i]] = 0
    # output the interestingness of the item:
    output_json_item_interesting_ness = {}
    for i in range(0, len(out_selected_item_names_list)):
        try:
            output_json_item_interesting_ness[out_selected_item_names_list[i]] = cmds.getAttr(out_selected_item_names_list[i]+".object_interestingness")
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
    global file_path
    # Code for loading the scene goes here
    text_file_path = cmds.textField(file_path, q=True, text=True)
    with open(text_file_path, "w") as f:
    # with open("/Users/evanpan/Documents/GitHub/Gaze_project/data/look_at_points/simplest_scene_fewer.json"):
        json.dump(output_json, f) 
    if look_through:
        current_panel = cmds.getPanel(wf = True)
        cmds.lookThru(current_panel, "persp")
    return out_selected_item_names_list, out_position_list

def load_scene(*args):
    global created_object_list
    global file_path
    # Code for loading the scene goes here
    text_file_path = cmds.textField(file_path, q=True, text=True)
    print(locals())
    if 'created_object_list' in globals():
        for i in range(0, len(created_object_list)):
            try:
                cmds.select(created_object_list[i])
                cmds.delete()
            except:
                print("cannot find object {}".format(i))
                created_object_list = []
    else:
        created_object_list = []
    default_file_text = "F:/MASC/Ribhav_processed_dataset/annotated_scene\Madelaine Petsch audition for The Prom.json"
    file_name = text_file_path
    file_content = json.load(open(file_name))
    # file_content = json.load(open("C:/Users/evansamaa/Documents/GitHub/Gaze_project/data/look_at_points/simplest_scene2_less_items.json"))
    temp_object_type, temp_object_pos, temp_object_interest = file_content["object_type"], file_content["object_pos"], file_content["object_interestingness"]
    for key in temp_object_type.keys():
        object_type = temp_object_type[key]
        object_pos = temp_object_pos[key]
        object_interest = temp_object_interest[key]
        sp, _ = cmds.sphere(r = 10, pivot =[0, 0, 0])
        cmds.select(sp)
        cmds.move(object_pos[0], object_pos[1], object_pos[2])
        created_object_list.append(sp)
        cmds.addAttr(sp, ln='object_type', at="float", k=True)
        cmds.setAttr(sp+".object_type", object_type)
        cmds.addAttr(sp, ln='object_interestingness', at="float", k=True)
        cmds.setAttr(sp+".object_interestingness", object_interest)
    print("load from: ", text_file_path)

# Define the save_scene function
def save_scene(*args):
    # Code for saving the scene goes here
    obtain_items_of_interest(True, False)
    print("Scene saved successfully!")

# Define the function for creating the UI window
def create_ui():
    window_name = "MyWindow"
    if cmds.window(window_name, exists=True):
        cmds.deleteUI(window_name)
    window = cmds.window(window_name, title="Load and Save Scene", widthHeight=(2000, 150))

    # Create two text boxes
    cmds.rowColumnLayout(numberOfColumns=1, columnWidth=[1, 1000])
    cmds.text(label="File path:")
    global file_path
    file_path = cmds.textField()

    # Create two buttons
    # cmds.rowColumnLayout(numberOfColumns=2, columnWidth=[(1, 200), (2, 200)])
    cmds.button(label="Load", command=load_scene)
    cmds.button(label="Save", command=save_scene)

    cmds.showWindow(window)

# Call the function to create the UI window
create_ui()