import json
import os
import shutil
import numpy as np
import gzip
import pickle as pkl
from Video_analysis_utils import get_wav_from_video
import cv2 as cv
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
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
def process_tobii_files(file_name):
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

    optical_flow_interp = compute_optical_flow_magnitude(os.path.join(input_dir, "fullstream.mp4"))

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
if __name__ == "__main__":
    # ========================================
    # ================ input =================
    # ========================================
    dt = 1.0 / 95
    threshold = 10
    # for file_name in os.listdir("F:/MASC/JALI_gaze/Tobii_recordings"):
    #     process_tobii_files(file_name)
    process_tobii_files("shakira")












