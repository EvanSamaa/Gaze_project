import pickle as pkl
import math

def load_neck_and_gaze(filename):
    neck_blend_dir = ["jNeck_xRotate", "jNeck_yRotate", "jNeck_zRotate"]
    for item in neck_blend_dir:
        try:
            cmds.delete(item + '_pairBlend')
        except:
            pass
        try:
            cmds.delete(item + '_alpha_speech')
        except:
            pass
        try:
            cmds.delete(item + '_speech')
        except:
            pass
        try:
            cmds.delete(item + '_gaze')
        except:
            pass

        cmds.createNode('pairBlend', n=item + '_pairBlend')
        cmds.createNode('animCurveTL', n=item + '_alpha_speech')        
        cmds.createNode('animCurveTL', n=item + '_speech')
        cmds.createNode('animCurveTL', n=item + '_gaze')
       
        cmds.connectAttr(item + '_speech.output', item + '_pairBlend.inTranslateX2')
        cmds.connectAttr(item + '_gaze.output', item + '_pairBlend.inTranslateX1')

    # this blends between two sets of musical articulation with pitch sensitivity

    motion_components = pkl.load(open(filename))
    ek, hk, ms, neck = motion_components
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
            cmds.setKeyframe("CNT_BOTH_EYES.LeftRight_eyes", t=interval[j][0] * 24, v=interval[j][1])
            cmds.setKeyframe("CNT_BOTH_EYES.DownUp_eyes", t=interval[j][0] * 24, v=-interval[j][2])
    
    for i in range(0, len(hk)):
        interval = hk[i]
        for j in range(0, len(interval)):
            # y direction has 
            cmds.setKeyframe("jNeck_ctl.jNeck_yRotate", t=interval[j][0] * 24, v=interval[j][1])
            cmds.setKeyframe("jNeck_ctl.jNeck_xRotate", t=interval[j][0] * 24, v=-interval[j][2])
    
    for i in range(0, len(ek)):
        interval = ek[i]
        for j in range(0, len(interval)):
            print(interval[j][1], 100 * math.tan(interval[j][1]/180 * 3.1415926))
            cmds.setKeyframe("eyeStare_world.eyeStare_world_translateX", time=interval[j][0]*24, value=interval[j][1])
            cmds.setKeyframe("eyeStare_world.eyeStare_world_translateY", time=interval[j][0]*24, value=interval[j][2])
            cmds.setKeyframe("eyeStare_world.eyeStare_world_translateZ", time=interval[j][0]*24, value=100)    
load_neck_and_gaze("C:/Users/evan1/Documents/Gaze_project/data/look_at_points/prototype2p2.pkl")