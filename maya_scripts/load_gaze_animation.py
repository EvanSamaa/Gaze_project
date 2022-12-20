import pickle as pkl
import math
def create_neck_blend_node():
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
    cmds.connectAttr('xneck_pairBlend.outTranslate.outTranslateX', 'jNeck_ctl.jNeck_xRotate')
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
    cmds.connectAttr('yneck_pairBlend.outTranslate.outTranslateX', 'jNeck_ctl.jNeck_yRotate')
    return
def load_gaze():
    create_neck_blend_node()
    filename = "C:/Users/evan1/Documents/Gaze_project/data/look_at_points/prototype2p2.pkl"
    filename = "C:/Users/evansamaa/Desktop/Gaze_project/data/look_at_points/prototype2p2.pkl"
    motion_components = pkl.load(open(filename))
    ek, hk, ms, neck = motion_components
    fps = mel.eval('float $fps = `currentTimeUnitToFPS`')
    try:
        cmds.cutKey("jNeck_ctl.jNeck_yRotate")
        cmds.cutKey("jNeck_ctl.jNeck_xRotate")
        cmds.cutKey("eyeStare_world.eyeStare_world_translateX")
        cmds.cutKey("eyeStare_world.eyeStare_world_translateY")
        cmds.cutKey("eyeStare_world.eyeStare_world_translateZ")
        cmds.cutKey("CNT_BOTH_EYES.LeftRight_eyes")
        cmds.cutKey("CNT_BOTH_EYES.DownUp_eyes")
    except:
        pass
    
    for i in range(0, len(ms)):
        interval = ms[i]
        for j in range(0, len(interval)):
            cmds.setKeyframe("CNT_BOTH_EYES.LeftRight_eyes", t=interval[j][0] * fps, v=interval[j][1])
            cmds.setKeyframe("CNT_BOTH_EYES.DownUp_eyes", t=interval[j][0] * fps, v=-interval[j][2])
    
    for i in range(0, len(hk)):
        interval = hk[i]
        for j in range(0, len(interval)):
            # y direction has
            cmds.setKeyframe("xneck_pairBlend_gaze", v=interval[j][1], t=interval[j][0] * fps)
            cmds.setKeyframe("yneck_pairBlend_gaze", v=-interval[j][2], t=interval[j][0] * fps)
            # cmds.setKeyframe("jNeck_ctl.jNeck_xRotate", t=interval[j][0] * fps, v=-interval[j][2])
    
    for i in range(0, len(ek)):
        interval = ek[i]
        for j in range(0, len(interval)):
            cmds.setKeyframe("eyeStare_world.eyeStare_world_translateX", time=interval[j][0] * fps, value=interval[j][1])
            cmds.setKeyframe("eyeStare_world.eyeStare_world_translateY", time=interval[j][0] * fps, value=interval[j][2])
            cmds.setKeyframe("eyeStare_world.eyeStare_world_translateZ", time=interval[j][0] * fps, value=interval[j][3])
    dims = ["x", "y", "z"]
    for k in range(0, 3):
        t = neck[k*2]
        v = neck[k*2+1]
        for i in range(len(t)):
            if dims[k] == "x" or dims[k] == "y":
                cmds.setKeyframe("{}neck_pairBlend_speech".format(dims[k]), v=-(v[i]),
                t=t[i] * fps)
            else:
                cmds.setKeyframe("jNeck_ctl.jNeck_{}Rotate".format(dims[k]), v=-(v[i]),
                                 t=t[i] * fps)
    
load_gaze()
