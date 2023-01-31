import json
import os


def clear_scene():
    return
   
def vicon_load_neck_relative_motion(vicon_path, tobii_path):
    fps = mel.eval('float $fps = `currentTimeUnitToFPS`')
    vicon_data = {}
    with open(vicon_path) as f:
        vicon_data = json.load(f)
    try:
        cmds.cutKey("jNeck_ctl.jNeck_yRotate")
        cmds.cutKey("jNeck_ctl.jNeck_xRotate")
        cmds.cutKey("jNeck_ctl.jNeck_zRotate")
        cmds.cutKey("Control_GRP.rotateX")
        cmds.cutKey("Control_GRP.rotateY")
        cmds.cutKey("Control_GRP.rotateZ")
        cmds.cutKey("Control_GRP.translateY")
        cmds.cutKey("Control_GRP.translateX")
        cmds.cutKey("Control_GRP.translateZ")
        cmds.cutKey("CNT_THOR.rotateX")
        cmds.cutKey("CNT_THOR.rotateY")
        cmds.cutKey("CNT_THOR.rotateZ")
    except:
        pass
    data = vicon_data["neck"]
    for i in range(0, len(data[0])):
        t = data[0][i]
        xyz = data[1][i]
        xyz = [f for f in xyz]
        z, y, x = xyz 
        cmds.setKeyframe("jNeck_ctl.jNeck_xRotate", v=x, t=t * fps)
        cmds.setKeyframe("jNeck_ctl.jNeck_yRotate", v=y, t=t * fps)
        cmds.setKeyframe("jNeck_ctl.jNeck_zRotate", v=z, t=t * fps)
    data_torso = vicon_data["root"]
    for i in range(0, len(data_torso[0])):
        t = data_torso[0][i]
        xyz = data_torso[1][i]
        xyz = [f for f in xyz]
        z, y, x = xyz 
        cmds.setKeyframe("CNT_THOR.rotateX", v=x, t=t * fps)
        cmds.setKeyframe("CNT_THOR.rotateY", v=y, t=t * fps)
        cmds.setKeyframe("CNT_THOR.rotateZ", v=z, t=t * fps)
    
    
def vicon_generate_objects_motion(vicon_path, tobii_path, distance_factor = 10, time_factor = 1, y_offset = 1630, z_offset=300):
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
        cmds.polyCube(w=10, h=12, d=14, n=key)
        cmds.move(x, y+y_offset, z+z_offset, key)
    # translate the objects
    for key in vicon_data.keys():
        data = vicon_data[key]
        for i in range(0, len(data[0])):
            t = data[0][i] / time_factor
            xyz = data[2][i]
            xyz = [f/distance_factor for f in xyz]
            x, z, y = xyz 
            cmds.setKeyframe(key+".translateX", v=x, t=t * fps)
            cmds.setKeyframe(key+".translateY", v=y+y_offset, t=t * fps)
            cmds.setKeyframe(key+".translateZ", v=z+z_offset, t=t * fps)
    # rotate the object
    for key in vicon_data.keys():
        data = vicon_data[key]
        for i in range(0, len(data[0])):
            t = data[0][i] / time_factor
            xyz = data[1][i]
            xyz = [f for f in xyz]
            x, z, y = xyz 
            cmds.setKeyframe(key+".rotateX", v=x, t=t * fps)
            cmds.setKeyframe(key+".rotateY", v=y+y_offset, t=t * fps)
            cmds.setKeyframe(key+".rotateZ", v=z+z_offset, t=t * fps)
    return vicon_data
def load_tobii(filename):
    # set the eye look-at-point to operate in local space
    cmds.setAttr("lookMaster.headWorldBlend", 0)
    fps = mel.eval('float $fps = `currentTimeUnitToFPS`')
    data = {}
    with open(filename) as f:
        data = json.load(f)
    gaze_data = data["gaze_local"]
    for j in range(0, len(gaze_data[0])):
        cmds.setKeyframe("eyeStare_head.eyeStare_head_translateX", time=gaze_data[0][j] * fps, value=gaze_data[1][j][0])
        cmds.setKeyframe("eyeStare_head.eyeStare_head_translateY", time=gaze_data[0][j] * fps, value=gaze_data[1][j][1])
        cmds.setKeyframe("eyeStare_head.eyeStare_head_translateZ", time=gaze_data[0][j] * fps, value=gaze_data[1][j][2]) 

        
cmds.currentUnit(l="mm")
# name_dict = vicon_generate_objects_motion("F:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/vicon_data.json", "")
# name_dict = vicon_generate_objects_motion("F:/MASC/JALI_gaze/Tobii_Vicon_recording/Misc_vicon_test/rotation_of_A_wrt_B_vicon.json", "")
output_path_relative = "D:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/vicon4_out_relative.json"
output_path_tobii = "D:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/tobii4/segments/1/aligned_livedata.json"
output_path = "D:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/vicon4_out.json"
vicon_load_neck_relative_motion(output_path, "")
load_tobii(output_path_tobii)
cmds.currentUnit(l="cm")


