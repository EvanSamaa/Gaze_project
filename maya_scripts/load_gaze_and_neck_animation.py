import pickle as pkl
import math
import json

def create_neck_addition_node(rig):
    try:
        cmds.cutKey("jNeck_ctl.jNeck_xRotate")
        cmds.cutKey("jNeck_ctl.jNeck_yRotate")
        cmds.cutKey("jNeck_ctl.jNeck_zRotate")
    except:
        pass
    try:        
        cmds.delete("jSync1_Vanilla_B1")
    except:
        pass
    try:
        cmds.delete("jSync1_Vanilla_B2")
    except:
        pass
    try:
        cmds.delete("jSync1_Vanilla_B3")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend_gaze")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend_speech")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend_blend")
    except:
        pass
    # make node for pairblend
    cmds.createNode('plusMinusAverage', n="xneck_pairBlend")
    # make node to store animation curves
    cmds.createNode('animCurveTL', n='xneck_pairBlend_gaze')
    cmds.createNode('animCurveTL', n='xneck_pairBlend_speech')
    cmds.createNode('animCurveTL', n='xneck_pairBlend_blend_speech1') # 1 is gaze, 0 is speech
    # make them connection
    # cmds.connectAttr('xneck_pairBlend_blend_speech1.output', 'xneck_pairBlend.weight')
    cmds.connectAttr('xneck_pairBlend_gaze.output', 'xneck_pairBlend.input1D[0]')
    cmds.connectAttr('xneck_pairBlend_speech.output', 'xneck_pairBlend.input1D[1]')
    cmds.connectAttr('xneck_pairBlend.output1D', 'jNeck_ctl.jNeck_xRotate')
    try:
        cmds.delete("yneck_pairBlend")
    except:
        pass
    try:
        cmds.delete("yneck_pairBlend_gaze")
    except:
        pass
    try:
        cmds.delete("yneck_pairBlend_speech")
    except:
        pass
    try:
        cmds.delete("yneck_pairBlend_blend")
    except:
        pass
    # make node for pairblend
    cmds.createNode('plusMinusAverage', n="yneck_pairBlend")
    # make node to store animation curves
    cmds.createNode('animCurveTL', n='yneck_pairBlend_gaze')
    cmds.createNode('animCurveTL', n='yneck_pairBlend_speech')
    cmds.createNode('animCurveTL', n='yneck_pairBlend_blend_speech1') # 1 is gaze, 0 is speech
    # make them connection
    # cmds.connectAttr('yneck_pairBlend_blend_speech1.output', 'yneck_pairBlend.weight')
    cmds.connectAttr('yneck_pairBlend_gaze.output', 'yneck_pairBlend.input1D[0]')
    cmds.connectAttr('yneck_pairBlend_speech.output', 'yneck_pairBlend.input1D[1]')
    cmds.connectAttr('yneck_pairBlend.output1D', 'jNeck_ctl.jNeck_yRotate')
    return

def create_neck_blend_node(rig):
    try:
        cmds.cutKey("jNeck_ctl.jNeck_xRotate")
        cmds.cutKey("jNeck_ctl.jNeck_yRotate")
        cmds.cutKey("jNeck_ctl.jNeck_zRotate")
    except:
        pass
    try:        
        cmds.delete("jSync1_Vanilla_B1")
    except:
        pass
    try:
        cmds.delete("jSync1_Vanilla_B2")
    except:
        pass
    try:
        cmds.delete("jSync1_Vanilla_B3")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend_gaze")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend_speech")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend_blend")
    except:
        pass
    # make node for pairblend
    cmds.createNode('pairBlend', n="xneck_pairBlend")
    # make node to store animation curves
    cmds.createNode('animCurveTL', n='xneck_pairBlend_gaze')
    cmds.createNode('animCurveTL', n='xneck_pairBlend_speech')
    cmds.createNode('animCurveTL', n='xneck_pairBlend_blend_speech1') # 1 is gaze, 0 is speech
    # make them connection
    cmds.connectAttr('xneck_pairBlend_blend_speech1.output', 'xneck_pairBlend.weight')
    cmds.connectAttr('xneck_pairBlend_gaze.output', 'xneck_pairBlend.inTranslateX1')
    cmds.connectAttr('xneck_pairBlend_speech.output', 'xneck_pairBlend.inTranslateX2')
    if rig == "jali":
        cmds.connectAttr('xneck_pairBlend.outTranslate.outTranslateX', 'jNeck_ctl.jNeck_xRotate')
    elif rig == "waitress":
        cmds.connectAttr('xneck_pairBlend.outTranslate.outTranslateX', 'neck_1_ctrl.rotateZ')
        
    try:
        cmds.delete("yneck_pairBlend")
    except:
        pass
    try:
        cmds.delete("yneck_pairBlend_gaze")
    except:
        pass
    try:
        cmds.delete("yneck_pairBlend_speech")
    except:
        pass
    try:
        cmds.delete("yneck_pairBlend_blend")
    except:
        pass
    # make node for pairblend
    cmds.createNode('pairBlend', n="yneck_pairBlend")
    # make node to store animation curves
    cmds.createNode('animCurveTL', n='yneck_pairBlend_gaze')
    cmds.createNode('animCurveTL', n='yneck_pairBlend_speech')
    cmds.createNode('animCurveTL', n='yneck_pairBlend_blend_speech1') # 1 is gaze, 0 is speech
    # make them connection
    cmds.connectAttr('yneck_pairBlend_blend_speech1.output', 'yneck_pairBlend.weight')
    cmds.connectAttr('yneck_pairBlend_gaze.output', 'yneck_pairBlend.inTranslateX1')
    cmds.connectAttr('yneck_pairBlend_speech.output', 'yneck_pairBlend.inTranslateX2')
    if rig == "jali":
        cmds.connectAttr('yneck_pairBlend.outTranslate.outTranslateX', 'jNeck_ctl.jNeck_yRotate')
    elif rig == "waitress":
        cmds.connectAttr('xneck_pairBlend.outTranslate.outTranslateX', 'neck_1_ctrl.yRotate')
    return
def load_tobii(filename, rig):
    fps = mel.eval('float $fps = `currentTimeUnitToFPS`')
    motion_components = pkl.load(open(filename))
    ek, hk = motion_components
    for i in range(0, len(hk)):
        interval = hk[i]
        for j in range(0, len(interval)):
            # x direction is vertical (elevation)
            cmds.setKeyframe("xneck_pairBlend_gaze", v=interval[j][2]/2, t=interval[j][0] * fps)
            # y direction is horizontal (azimuth) 
            cmds.setKeyframe("yneck_pairBlend_gaze", v=interval[j][1]/2, t=interval[j][0] * fps)
            # cmds.setKeyframe("jNeck_ctl.jNeck_xRotate", t=interval[j][0] * fps, v=-interval[j][2])
    
    for i in range(0, len(ek)):
        interval = ek[i]
        for j in range(0, len(interval)):
            if rig == "jali":
                cmds.setAttr("lookMaster.headWorldBlend", 0)
                cmds.setKeyframe("eyeStare_head.eyeStare_head_translateX", time=interval[j][0] * fps, value=interval[j][1])
                cmds.setKeyframe("eyeStare_head.eyeStare_head_translateY", time=interval[j][0] * fps, value=interval[j][2])
                cmds.setKeyframe("eyeStare_head.eyeStare_head_translateZ", time=interval[j][0] * fps, value=interval[j][3])
            elif rig == "waitress":
                cmds.setKeyframe("Eye_aim_ctrls.translateX", time=interval[j][0] * fps, value=interval[j][1])
                cmds.setKeyframe("Eye_aim_ctrls.translateY", time=interval[j][0] * fps, value=interval[j][2])
                cmds.setKeyframe("Eye_aim_ctrls.translateZ", time=interval[j][0] * fps, value=interval[j][3])
def load_video_annotation(filename, motion_to_load):
    fps = mel.eval('float $fps = `currentTimeUnitToFPS`')
    try:
        if rig == "jali":
            cmds.cutKey("jNeck_ctl.jNeck_yRotate")
            cmds.cutKey("jNeck_ctl.jNeck_xRotate")
            cmds.cutKey("eyeStare_world.eyeStare_world_translateX")
            cmds.cutKey("eyeStare_world.eyeStare_world_translateY")
            cmds.cutKey("eyeStare_world.eyeStare_world_translateZ")
            cmds.cutKey("CNT_BOTH_EYES.LeftRight_eyes")
            cmds.cutKey("CNT_BOTH_EYES.DownUp_eyes")
        elif rig == "waitress":
            cmds.cutKey("neck_1_ctrl.rotateY")
            cmds.cutKey("neck_1_ctrl.rotateZ")
            cmds.cutKey("neck_1_ctrl.rotateX")
            cmds.cutKey("Eye_aim_ctrls.translateX")
            cmds.cutKey("Eye_aim_ctrls.translateY")
            cmds.cutKey("Eye_aim_ctrls.translateZ")
            cmds.cutKey("R_eye_aim_ctrl.translateX")
            cmds.cutKey("R_eye_aim_ctrl.translateY")
            cmds.cutKey("L_eye_aim_ctrl.translateX")
            cmds.cutKey("L_eye_aim_ctrl.translateY")
    except:
        pass
    motion = json.load(open(filename))
    ts = motion["ts"]
    x, y = motion[motion_to_load]
    for j in range(0, len(x)):
        # x direction is vertical (elevation)
        cmds.setKeyframe("jNeck_ctl.jNeck_xRotate", v=-x[j], t=ts[j] * fps)
        # y direction is horizontal (azimuth) 
        cmds.setKeyframe("jNeck_ctl.jNeck_yRotate", v=y[j], t=ts[j] * fps)
    
    
def load_gaze(filename, rig, tobii = False):
    # create_neck_blend_node(rig)
    create_neck_addition_node(rig)
    # filename = "C:/Users/evan1/Documents/Gaze_project/data/look_at_points/prototype2p2.pkl"
    # filename = "C:/Users/evansamaa/Desktop/Gaze_project/data/look_at_points/prototype2p2.pkl"
    fps = mel.eval('float $fps = `currentTimeUnitToFPS`')
    try:
        if rig == "jali":
            cmds.cutKey("jNeck_ctl.jNeck_yRotate")
            cmds.cutKey("jNeck_ctl.jNeck_xRotate")
            cmds.cutKey("eyeStare_world.eyeStare_world_translateX")
            cmds.cutKey("eyeStare_world.eyeStare_world_translateY")
            cmds.cutKey("eyeStare_world.eyeStare_world_translateZ")
            cmds.cutKey("CNT_BOTH_EYES.LeftRight_eyes")
            cmds.cutKey("CNT_BOTH_EYES.DownUp_eyes")
        elif rig == "waitress":
            cmds.cutKey("neck_1_ctrl.rotateY")
            cmds.cutKey("neck_1_ctrl.rotateZ")
            cmds.cutKey("neck_1_ctrl.rotateX")
            cmds.cutKey("Eye_aim_ctrls.translateX")
            cmds.cutKey("Eye_aim_ctrls.translateY")
            cmds.cutKey("Eye_aim_ctrls.translateZ")
            cmds.cutKey("R_eye_aim_ctrl.translateX")
            cmds.cutKey("R_eye_aim_ctrl.translateY")
            cmds.cutKey("neck_pairBlend_speechL_eye_aim_ctrl.translateX")
            cmds.cutKey("L_eye_aim_ctrl.translateY")
    except:
        pass
    if tobii:
        return load_tobii(filename, rig)
    motion_components = pkl.load(open(filename))
    ek = motion_components["eye_frames"]
    hk = motion_components["head_frames"]
    ms = motion_components["micro_saccade"]
    neck = motion_components["other_neck"]
    envelope = motion_components["envelope"]
    
    
    
    
    cmds.setAttr("lookMaster.headWorldBlend", 10)
    # micro-saccade
    for i in range(0, len(ms)):
        interval = ms[i]
        for j in range(0, len(interval)):
            if rig == "jali":
                cmds.setKeyframe("CNT_BOTH_EYES.LeftRight_eyes", t=interval[j][0] * fps, v=interval[j][1])
                cmds.setKeyframe("CNT_BOTH_EYES.DownUp_eyes", t=interval[j][0] * fps, v=-interval[j][2])
            elif rig == "waitress":
                cmds.setKeyframe("R_eye_aim_ctrl.translateX", t=interval[j][0] * fps, v=interval[j][1])
                cmds.setKeyframe("R_eye_aim_ctrl.translateY", t=interval[j][0] * fps, v=-interval[j][2])
                cmds.setKeyframe("L_eye_aim_ctrl.translateX", t=interval[j][0] * fps, v=interval[j][1])
                cmds.setKeyframe("L_eye_aim_ctrl.translateY", t=interval[j][0] * fps, v=-interval[j][2])
    
    for i in range(0, len(hk)):
        interval = hk[i]
        for j in range(0, len(interval)):
            # x direction is vertical (elevation)
            cmds.setKeyframe("xneck_pairBlend_gaze", v=-interval[j][2], t=interval[j][0] * fps)
            # y direction is horizontal (azimuth) 
            cmds.setKeyframe("yneck_pairBlend_gaze", v=interval[j][1], t=interval[j][0] * fps)
            # cmds.setKeyframe("jNeck_ctl.jNeck_xRotate", t=interval[j][0] * fps, v=-interval[j][2])
    
    for i in range(0, len(ek)):
        interval = ek[i]
        for j in range(0, len(interval)):
            if rig == "jali":
                cmds.setKeyframe("eyeStare_world.eyeStare_world_translateX", time=interval[j][0] * fps, value=interval[j][1])
                cmds.setKeyframe("eyeStare_world.eyeStare_world_translateY", time=interval[j][0] * fps, value=interval[j][2])
                cmds.setKeyframe("eyeStare_world.eyeStare_world_translateZ", time=interval[j][0] * fps, value=interval[j][3])
            elif rig == "waitress":
                cmds.setKeyframe("Eye_aim_ctrls.translateX", time=interval[j][0] * fps, value=interval[j][1])
                cmds.setKeyframe("Eye_aim_ctrls.translateY", time=interval[j][0] * fps, value=interval[j][2])
                cmds.setKeyframe("Eye_aim_ctrls.translateZ", time=interval[j][0] * fps, value=interval[j][3])
    dims = ["x", "y", "z"]
    cmds.setKeyframe("xneck_pairBlend_blend_speech1", v=0, t=0)
    cmds.setKeyframe("yneck_pairBlend_blend_speech1", v=0, t=0)
    for k in range(0, 3):
        t = neck[k*2]
        v = neck[k*2+1]
        for i in range(len(t)):
            if dims[k] == "x" or dims[k] == "y":
                cmds.setKeyframe("{}neck_pairBlend_speech".format(dims[k]), v=-(v[i]),
                t=t[i] * fps)
            else:
                if rig == "jali":
                    cmds.setKeyframe("jNeck_ctl.jNeck_{}Rotate".format(dims[k]), v=(v[i]),
                                 t=t[i] * fps)
                elif rig == "waitress":
                    cmds.setKeyframe("neck_1_ctrl.RotateZ".format(dims[k]), v=-(v[i]),
                                 t=t[i] * fps)
        
# load_video_annotation('/Users/evanpan/Documents/GitHub/Gaze_project/data/look_at_points/video_annotation.json', "all_head")
# load_gaze("C:/Users/evansamaa/Desktop/Gaze_project/data/prototype2p2.pkl", "jali")
# load_gaze("C:/Users/evansamaa/Desktop/Gaze_project/data/tobii_data/shakira/tobii_rotation.pkl", "jali", True)
load_gaze("F:/MASC/Ribhav_processed_dataset/outputs\Madelaine Petsch audition for The Prom.pkl", "jali")
