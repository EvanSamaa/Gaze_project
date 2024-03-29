import json
import pickle as pkl
if 'created_object_list' in locals():
    for i in range(0, len(created_object_list)):
        try:
            cmds.select(created_object_list[i])
            cmds.delete()
        except:
            print("cannot find object {}".format(created_object_list[i]))
            created_object_list = []
else:
    created_object_list = []
    
file_name = "F:/MASC/Ribhav_processed_dataset/annotated_scene\Madelaine Petsch audition for The Prom.json"
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
            
            