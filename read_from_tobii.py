import json
import os
import sys
sys.path.insert(0, "C:/Users/evansamaa/Desktop/staggered_face/Utils/")
import shutil
import numpy as np
import gzip
import pickle as pkl
import scipy.signal
from Video_analysis_utils import get_wav_from_video
import cv2 as cv
from Signal_processing_utils import dx_dt
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from Geometry_Util import rotation_matrix_from_vectors
import csv
import re
import json
import math
from scipy.spatial.transform import Rotation
def compute_optical_flow_magnitude(filename):
    cap = cv.VideoCapture(filename)
    fps = cap.get(cv.CAP_PROP_FPS)
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_RGB2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    out_x = [0]
    while(1):
    # while(len(out_x) < 5):
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        next = cv.cvtColor(frame2, cv.COLOR_RGB2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        avg_mag = np.square(flow).sum(axis=2).mean()
        out_x.append(avg_mag)
        prvs = next
    out_t = np.arange(0, len(out_x))/fps
    return interp1d(out_t, out_x, bounds_error=False, fill_value="extrapolate")
def process_tobii_files_for_maya(file_name):
    input_dir = "F:/MASC/JALI_gaze/Tobii_recordings/{}/segments/1/".format(file_name)
    output_dir = "./data/tobii_data/{}/".format(file_name)

    # ========================================
    # ================= code =================
    # ========================================
    try:
        os.mkdir(output_dir)
    except:
        pass
    # move file
    shutil.copyfile(os.path.join(input_dir, "fullstream.mp4"), os.path.join(output_dir, "fullstream.mp4"))
    # extract file
    with gzip.open(os.path.join(input_dir, "livedata.json.gz"), 'rb') as f_in:
        with open(os.path.join(output_dir, "livedata.json"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    # analyaze data
    input = []
    with open(os.path.join(output_dir, "livedata.json"), "rb") as f:
        lines = f.readlines()
    for l in lines:
        input.append(json.loads(l))
    # sort by time stamp
    input = sorted(input, key=lambda x: x["ts"])
    prev_gaze = input[0]
    prev_gyro = input[0]

    # store the angular velocity of the gyro scope reading
    gyro_t = []
    gyro_rotation_angular_speed = []
    gyro_rotation_angle = [[0, 0, 0]]

    # store the viewing position
    gaze_t = []
    gaze_pos = []

    # compute the gaze positions over time
    for i in range(len(input)):
        # skip repeated time stamps for gaze
        if input[i]["ts"] == prev_gaze["ts"]:
            pass
        else:
            try:
                # print("gaze", input[i]["ts"], input[i]["gp3"], input[i]["ts"] - prev_gaze["ts"])
                gaze_pos.append(input[i]["gp3"])
                gaze_t.append(input[i]["ts"] - input[0]["ts"])
                prev_gaze = input[i]
            except:
                pass
            try:
                # print("gyro", input[i]["ts"], input[i]["gy"], input[i]["ts"] - prev_gyro["ts"])
                gyro_rotation_angular_speed.append(input[i]["gy"])
                gyro_t.append(input[i]["ts"] - input[0]["ts"])
                prev_gyro = input[i]
            except:
                pass

    ###############################################################################################################
    ###################### compute the neck rotations over time from gyroscopic data and video ####################
    ###############################################################################################################

    # optical_flow_interp = compute_optical_flow_magnitude(os.path.join(input_dir, "fullstream.mp4"))

    for i in range(1, len(gyro_rotation_angular_speed)):
        prev = gyro_rotation_angle[i - 1]
        dR_dt = gyro_rotation_angular_speed[i]
        dR_dt[0] = gyro_rotation_angular_speed[i][0] if optical_flow_interp(i*dt) > threshold else 0
        dR_dt[1] = gyro_rotation_angular_speed[i][1] if optical_flow_interp(i*dt) > threshold else 0
        dR_dt[2] = gyro_rotation_angular_speed[i][2] if optical_flow_interp(i*dt) > threshold else 0

        current = [prev[0] + dR_dt[0] * dt,
                   prev[1] + dR_dt[1] * dt,
                   prev[2] + dR_dt[2] * dt]
        gyro_rotation_angle.append(current)

    out_gaze = []
    out_neck = []

    for i in range(0, len(gaze_t)):
        frame = [gaze_t[i] / 1000000, gaze_pos[i][0], gaze_pos[i][1], gaze_pos[i][2]]
        out_gaze.append(frame)
    for i in range(0, len(gyro_rotation_angle)):
        frame = [gyro_t[i] / 950000, gyro_rotation_angle[i][0], gyro_rotation_angle[i][1], gyro_rotation_angle[i][2]]
        out_neck.append(frame)
    out = [[out_gaze], [out_neck]]
    pkl.dump(out, open(os.path.join(output_dir, "tobii_rotation.pkl".format(file_name)), "wb"), protocol=2)
    get_wav_from_video("fullstream.mp4", output_dir)
def read_csv_header(csvfile):
    """
    Parse the header of a Vicon csv file.
    Returns list of tracked object names.
    """
    # Skip garbage object count
    csvfile.readline()
    csvfile.readline()

    csvreader = csv.reader(csvfile)
    object_name_row = next(csvreader, None)

    # Skip column labels
    next(csvreader)
    next(csvreader)

    # Also remove the "Global Angle" prefix from the object names
    name_matcher = re.compile(r"\w+:\w+")
    names = []

    for i in range(2, len(object_name_row) - 1, 6):
        match = name_matcher.search(object_name_row[i])
        if match is not None:
            names.append(match.group())
    return names
def load_vicon_file(file_name, target_sr=100, neutral_frame = 0):
    csv_file = open(file_name)
    # get the list of objects in the scene
    objects_names = read_csv_header(csv_file)
    print(objects_names)
    num_of_objects = len(objects_names)    
    data = csv_file.readlines()
    # output dictionary
    out_dict = {} 
    # this dictionary will have shape {name: [[time], [rotations], [positions]]}
    for item in range(num_of_objects):
        out_dict[objects_names[item]] = [[], [], []] # these are for [time], [positions], and [rotations] respectively
    # load up data
    for i in range(0, len(data)):
        line_list = (data[i].strip("\n")).split(",") # get all the data
        # the final line might have no data
        if len(line_list) < 2: break 
        # get the frame number
        time = float(line_list[0])
        # get the position and rotation
        for j in range(0, num_of_objects):
            item = objects_names[j]
            start = 2 + j*6
            end = start + 6
            data_cols = line_list[start:end]
            if data_cols[0] == "":
                continue
            else:
                # get rotations and positions
                out_dict[item][0].append(time/100)
                rotations = data_cols[:3]
                rotations = [float(i)*180.0/math.pi for i in rotations]
                out_dict[item][1].append(rotations)
                positions = data_cols[3:]
                positions = [float(i) for i in positions]
                out_dict[item][2].append(positions)
    return out_dict
def get_relatave_rotation(file_name, root_name, child_name, target_sr=100):
    vicon_data = load_vicon_file(file_name, target_sr)
    root_configuration = vicon_data[root_name]
    r_positions = np.array(root_configuration[2])
    r_rotations = np.array(root_configuration[1])    
    child_configuration = vicon_data[child_name]
    c_positions = np.array(child_configuration[2])
    c_rotations = np.array(child_configuration[1])

    # get the last timestamp in frames
    last_vicon_frame = np.maximum(root_configuration[0][-1], child_configuration[0][-1])
    # get the last timestamp in time (we know it's 100 fps)
    last_vicon_time = last_vicon_frame
    # get a new range for the rotations     
    new_time_range = np.arange(0, last_vicon_time, 1/target_sr)    
    # make interpolation matches the new sampling rate
    r_rotations_interp = interp1d(np.array(root_configuration[0]), r_rotations, bounds_error=False, fill_value=(r_rotations[0], r_rotations[-1]), axis=0)
    c_rotations_interp = interp1d(np.array(child_configuration[0]), c_rotations, bounds_error=False, fill_value=(c_rotations[0], c_rotations[-1]), axis=0)
    r_rotations = r_rotations_interp(new_time_range)
    c_rotations = c_rotations_interp(new_time_range)
    r_positions_interp = interp1d(np.array(root_configuration[0]), r_positions, bounds_error=False, fill_value=(r_positions[0], r_positions[-1]), axis=0)
    r_positions = r_positions_interp(new_time_range)
    r_positions = r_positions - r_positions[0]
    # get them as rotation matrices
    root_R = Rotation.from_euler("xyz",r_rotations,degrees=True).as_matrix()
    child_R = Rotation.from_euler("xyz",c_rotations,degrees=True).as_matrix()
    root_R = np.linalg.inv(root_R[600:601]) @ root_R
    child_R = np.linalg.inv(child_R[600:601]) @ child_R
    # back_prop the rotation to find the 
    child_local_R = np.linalg.inv(root_R) @ child_R
    c_rotations_relative = Rotation.from_matrix(child_local_R).as_euler('xyz', degrees=True)
    c_rotations_global = Rotation.from_matrix(child_R).as_euler('xyz', degrees=True)
    root_R = Rotation.from_matrix(root_R).as_euler('xyz', degrees=True)
    output_json = {}
    plt.plot(r_positions)
    plt.show()
    output_json["neck"] = [new_time_range.tolist(), c_rotations_global.tolist(), c_rotations_relative.tolist()]
    output_json["root"] = [new_time_range.tolist(), root_R.tolist(), r_positions.tolist()]
    return output_json    
def load_tobii_file(file_name, target_sr=100):
    
    # see if the file has been decompressed already
    decompressed_file_name = file_name.split(".")[0]+".json"
    if not os.path.exists(decompressed_file_name):        
        with gzip.open(file_name) as f_in:
            with open(decompressed_file_name, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    # load json file
    tobii_data = []
    with open(decompressed_file_name, "rb") as f:
        lines = f.readlines()
    for l in lines:
        tobii_data.append(json.loads(l))
    # sort by time stamp
    tobii_data = sorted(tobii_data, key=lambda x: x["ts"])
    # get the time_stamp/gaze_direction out
    gaze_t = [] # timestamp for the gaze x, y, z
    gaze_v = [] # x, y, z coordinate for the gaze
    gyro_t = [] # timestamp for the gyro reading
    gyro_v = [] # angular velocity of tobii in x, y, z
    ts = [] # time code for tobii
    vts = [] # time code for video    
    
    for i in range(0, len(tobii_data)):        
        try:
            # print("gaze", input[i]["ts"], input[i]["gp3"], input[i]["ts"] - prev_gaze["ts"])
            gaze_v.append(tobii_data[i]["gp3"])
            gaze_t.append(tobii_data[i]["ts"])
        except:
            pass
        try:
            # print("gyro", input[i]["ts"], input[i]["gy"], input[i]["ts"] - prev_gyro["ts"])
            gyro_v.append(tobii_data[i]["gy"])
            gyro_t.append(tobii_data[i]["ts"])
        except:
            pass
        try:
            vts.append(tobii_data[i]["vts"])
            ts.append(tobii_data[i]["ts"])
        except:
            pass
    gaze_t = np.array(gaze_t)
    gaze_t = gaze_t - ts[0]
    gaze_v = np.array(gaze_v)
    gyro_t = np.array(gyro_t)
    gyro_t = gyro_t - ts[0]
    gyro_v = np.array(gyro_v)
    ts = np.array(ts)
    vts = np.array(vts)
    # it is known that the glasses operate at 50hz, therefore each dt corresponds to 20ms
    dt_gaze = (gaze_t[1:] - gaze_t[0:-1]).mean()
    # the conversion rate from sampling rate to seconds
    conversion_rate = (1/50) / dt_gaze    
    gaze_t = gaze_t * conversion_rate
    gyro_t = gyro_t * conversion_rate
    # make a new sampling rate for 100 Hz to align both gaze and gyro data
    upper_limit = np.round(gaze_t[-1] * 100) / 100
    new_time_range = np.arange(0, upper_limit, 1/target_sr)
    aligned_gaze = interp1d(gaze_t, gaze_v, bounds_error=False, axis=0)(new_time_range)
    aligned_gyro = interp1d(gyro_t, gyro_v, bounds_error=False, axis=0)(new_time_range)
    output_json = {"gaze_local": [new_time_range.tolist(), aligned_gaze.tolist()],
                "gyro": [new_time_range.tolist(), aligned_gyro.tolist()]}
    return output_json
def align_tobii_vicon(out_dict_tobii, out_dict_vicon):
    # get the head rotation angle from the head
    vicon_head_rotation_angle = np.array(out_dict_vicon["neck"][1])
    # the the angular velocity (by computing dx_dt)
    vicon_head_gyro = dx_dt(vicon_head_rotation_angle[:, 2], 0.01)
    tobii_head_gyro = np.array(out_dict_tobii["gyro"][1])[:, 1]
    # compute delay array using correlation
    delay = scipy.signal.correlate(vicon_head_gyro, tobii_head_gyro, mode="valid")
    # compute the peak delay
    peak_delay = np.argmax(delay)
    out_dict_vicon_aligned = {}
    new_start = peak_delay

    out_dict_vicon_aligned["neck"] = []
    out_dict_vicon_aligned["neck"].append(out_dict_vicon["neck"][0][new_start:])
    out_dict_vicon_aligned["neck"][0] = [j - out_dict_vicon_aligned["neck"][0][0] for j in out_dict_vicon_aligned["neck"][0]]
    for i in range(1, len(out_dict_vicon["neck"])):
        out_dict_vicon_aligned["neck"].append(out_dict_vicon["neck"][i][new_start:])
    out_dict_vicon_aligned["root"] = []
    out_dict_vicon_aligned["root"].append(out_dict_vicon["root"][0][new_start:])
    out_dict_vicon_aligned["root"][0] = [j - out_dict_vicon_aligned["root"][0][0] for j in
                                         out_dict_vicon_aligned["root"][0]]
    for i in range(1, len(out_dict_vicon["root"])):
        out_dict_vicon_aligned["root"].append(out_dict_vicon["root"][i][new_start:])
    return out_dict_vicon_aligned

if __name__ == "__main__":
    # ========================================
    # ================ input =================
    # ========================================
    # vicon
    input_path = "D:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/vicon4.csv"
    # tobii
    input_tobii = "D:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/tobii4/segments/1/livedata.json.gz"
    # output_path
    output_path_relative = "D:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/vicon4_out_relative.json"
    output_path_tobii = "D:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/tobii4/segments/1/aligned_livedata.json"
    output_path = "D:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/vicon4_out.json"

    # output_path = "F:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/vicon_data.json"
    # input_path = "F:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/vicon.csv"

    # for trial 1-3 recorded with Vicon Tracker
    goal = "load_motion"
    if goal == "load_motion" or goal == "all":
        # get the json object
        out_dict_vicon = load_vicon_file(input_path)
        # json the output dictionary and save it to file 
        json_object = json.dumps(out_dict_vicon, indent=4)
        with open(output_path, "w") as f:
            f.write(json_object)
    if goal == "relative_rotation" or goal == "all":
        # out_dict_relative = get_relatave_rotation(input_path, "Torsooo:Torsooo", "glasses:glasses")
        out_dict_relative = get_relatave_rotation(input_path, "torso:torso", "glasses:glasses")
        json_object = json.dumps(out_dict_relative, indent=4)
        with open(output_path_relative, "w") as f:
            f.write(json_object)
    if goal == "load_tobii" or goal == "all":
        out_dict_tobii = load_tobii_file(input_tobii)
        json_object = json.dumps(out_dict_tobii, indent=4)
        with open(output_path_tobii, "w") as f:
            f.write(json_object)
    if goal == "all":
        new_out_dict_relative = align_tobii_vicon(out_dict_tobii, out_dict_relative)
        json_object = json.dumps(new_out_dict_relative, indent=4)
        with open(output_path_relative, "w") as f:
            f.write(json_object)
    # new_time_range, aligned_gaze, aligned_gyro = load_tobii_file("F:/MASC/JALI_gaze/Tobii_Vicon_recording/Integration_test/tobii/segments/1/livedata.json.gz")
    # print(aligned_gaze.shape)
    out_dict_tobii = json.load(open(output_path_tobii))
    out_dict_vicon = json.load(open(output_path_relative))
    new_out_dict_vicon =  align_tobii_vicon(out_dict_tobii, out_dict_vicon) 
    












