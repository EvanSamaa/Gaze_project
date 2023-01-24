import json
import os


def clear_scene():
    return
   

def vicon_generate_objects_motion(vicon_path, tobii_path, distance_factor = 100, time_factor = 120):
    fps = mel.eval('float $fps = `currentTimeUnitToFPS`')
    # get the vicon data
    vicon_data = {}
    with open(vicon_path) as f:
        vicon_data = json.load(f)
    # remove existing versions of these objects
    for key in vicon_data.keys():
        try:
            cmds.delete(key)
        except:
            pass
    print(vicon_data.keys())    
    # create the objects
    for key in vicon_data.keys():
        initial_position_data = vicon_data[key][2][0]
        initial_position_data = [f/distance_factor for f in initial_position_data]
        x, z, y = initial_position_data 
        # create a sphere
        cmds.sphere(r=1, n=key)
        cmds.move(x, y, z, key)
    # move the objects
    for key in vicon_data.keys():
        data = vicon_data[key]
        for i in range(0, len(data[0])):
            t = data[0][i] / time_factor
            xyz = data[2][i]
            xyz = [f/distance_factor for f in xyz]
            x, z, y = xyz 
            cmds.setKeyframe(key+".translateX", v=x, t=t * fps)
            cmds.setKeyframe(key+".translateY", v=y, t=t * fps)
            cmds.setKeyframe(key+".translateZ", v=z, t=t * fps)
    return vicon_data

        

cmds.currentUnit(l="mm")
name_dict = vicon_generate_objects_motion("F:/MASC/JALI_gaze/Tobii_Vicon_recording/Misc_vicon_test/rotation_of_A_wrt_B_vicon.json", "")

cmds.currentUnit(l="cm")

