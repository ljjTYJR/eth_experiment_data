# from asyncio.windows_events import NULL
from cProfile import label
from cmath import pi
import imp
from statistics import mode
# from typing import Protocol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# used to plot a smooth curve
from scipy.interpolate import make_interp_spline
import os


# Define the file path
    # The Result path
eth = 0
stairs = 1
apartment = 2
plain = 3
wood = 4
gazebo = 5

Path = {
    eth : "/media/shuo/T7/ETH/eth/",
    stairs : "/media/shuo/T7/ETH/Stairs/",
    apartment : "/media/shuo/T7/ETH/apartment/apartment_local/",
    plain : "/media/shuo/T7/ETH/plain/",
    wood : "/media/shuo/T7/ETH/wood_summer/",
    gazebo : "/media/shuo/T7/ETH/gazebo_winter/"
}

Result_file = "_result.txt"

Validation_file = {
    eth : "eth_validation.csv",
    stairs : "stairs_validation.csv",
    apartment : "apartment_validation.csv",
    plain : "plain_validation.csv",
    wood : "wood_validation.csv",
    gazebo : "gazebo_validation.csv"
}

labels = {
    eth : 'eth',
    stairs : 'stairs',
    apartment : 'apartment',
    plain : 'plain',
    wood : 'wood',
    gazebo : 'gazebo'
}

# show results with high overlap
def show_pose_result(type):

    # Get data from the file
    result_dict = {}
    validation_dict = {}
    for i in range(eth, gazebo + 1, 1):
        if os.path.exists(labels[i] + Result_file):
            result_dict[i] = np.loadtxt(labels[i] + Result_file, float, delimiter=',', skiprows=0)

        if os.path.exists(Path[i] + Validation_file[i]):
            tmp = np.loadtxt(Path[i] + Validation_file[i], str, delimiter=',', skiprows=1)
            validation_dict[i] = tmp[::10]
    LIST_LEN = len(result_dict[apartment])

    # print("translation error:", result_dict[stairs][:, 1] / math.pi * 180)
    for i in range(eth, gazebo + 1, 1):
        if i in result_dict:
            if (type == 'easy'):
                result_dict[i] = result_dict[i][validation_dict[i][:, 1] == ' easyPoses']
            elif (type == 'medium'):
                result_dict[i] = result_dict[i][validation_dict[i][:, 1] == ' mediumPoses']
            elif (type == 'hard'):
                result_dict[i] = result_dict[i][validation_dict[i][:, 1] == ' hardPoses']

            # print(len(result_dict[i]))

    LIST_LEN = len(result_dict[apartment])
    TRANS_THRESHOLD_MIN = 0.0
    TRANS_THRESHOLD_MAX = 4.0
    TRANS_STEP = 0.01
    TRANS_NUM = int((TRANS_THRESHOLD_MAX + 2 * TRANS_STEP - TRANS_THRESHOLD_MIN) / TRANS_STEP)
    TRANS_POINT_NUM = TRANS_NUM * 2
    translation_error_cumulative_arr = {}
    for i in range(eth, gazebo + 1, 1):
        # NOTE: Here, the maximum value is modified to the maximum data;
        TRANS_THRESHOLD_MIN = 0.0
        TRANS_THRESHOLD_MAX = max(result_dict[i][:, 1])
        TRANS_STEP = 0.01
        TRANS_NUM = int((TRANS_THRESHOLD_MAX + 2 * TRANS_STEP - TRANS_THRESHOLD_MIN) / TRANS_STEP)
        TRANS_POINT_NUM = TRANS_NUM * 2
        if i in result_dict:
            translation_error_cumulative_arr[i] = []
            for j in range(TRANS_NUM):
                current_threshold = float(j * TRANS_STEP)
                cur_probability = sum(1 for val in result_dict[i][:, 1] if val <= current_threshold) / LIST_LEN
                translation_error_cumulative_arr[i].append([current_threshold, cur_probability])
            translation_error_cumulative_arr[i] = np.array(translation_error_cumulative_arr[i]) # change it to array
            if not os.path.exists(type):
                os.mkdir(type)
            np.savetxt(type + '/' + labels[i] + "_cumulative_translation_probability", translation_error_cumulative_arr[i], fmt="%.5f", delimiter=',')


    # Create the model dict to store points
    translation_model_dict = {}
    x_sim_trans = np.linspace(TRANS_THRESHOLD_MIN, TRANS_THRESHOLD_MAX, TRANS_POINT_NUM)
    for i in range(eth, gazebo + 1, 1):
        if i in translation_error_cumulative_arr:
            trans_model = make_interp_spline(translation_error_cumulative_arr[i][:, 0], translation_error_cumulative_arr[i][:, 1])
            y_sim_trans = trans_model(x_sim_trans)
            translation_model_dict[i] = y_sim_trans

    ROT_THRESHOLD_MIN = 0.0
    ROT_THRESHOLD_MAX = 25.0
    ROT_STEP = 0.01
    ROT_NUM = int((ROT_THRESHOLD_MAX + 2 * ROT_STEP - ROT_THRESHOLD_MIN) / ROT_STEP)
    ROT_POINT_NUM = ROT_NUM * 2
    # Convert the angles to degrees
    rotation_error_deg_dict = {}
    for i in range(eth, gazebo + 1, 1):
        if i in result_dict:
            ROT_THRESHOLD_MIN = 0.0
            ROT_THRESHOLD_MAX = max(result_dict[i][:, 2]) / math.pi * 180
            ROT_STEP = 0.01
            ROT_NUM = int((ROT_THRESHOLD_MAX + 2 * ROT_STEP - ROT_THRESHOLD_MIN) / ROT_STEP)
            ROT_POINT_NUM = ROT_NUM * 2
            rotation_error_deg_dict[i] = result_dict[i][:, 2] / math.pi * 180

    rotation_error_cumulative_arr = {}
    for i in range(eth, gazebo + 1, 1):
        if i in result_dict:
            rotation_error_cumulative_arr[i] = []
            for j in range(ROT_NUM):
                current_threshold = float(j * ROT_STEP)
                cur_probability = sum(1 for val in rotation_error_deg_dict[i] if val <= current_threshold) / LIST_LEN
                rotation_error_cumulative_arr[i].append([current_threshold, cur_probability])
            rotation_error_cumulative_arr[i] = np.array(rotation_error_cumulative_arr[i])
            if not os.path.exists(type):
                os.mkdir(type)
            np.savetxt(type + '/' + labels[i] + "_cumulative_rotation_probability", rotation_error_cumulative_arr[i], fmt="%.5f", delimiter=',')

    # print(rotation_error_cumulative_arr[apartment])
    rotation_model_dict = {}
    x_sim_rotation = np.linspace(ROT_THRESHOLD_MIN, ROT_THRESHOLD_MAX, ROT_POINT_NUM)
    for i in range(eth, gazebo + 1, 1):
        if i in rotation_error_cumulative_arr:
            trans_model = make_interp_spline(rotation_error_cumulative_arr[i][:, 0], rotation_error_cumulative_arr[i][:, 1])
            y_sim_rotation = trans_model(x_sim_rotation)
            rotation_model_dict[i] = y_sim_rotation

    """
    " Just for the data
    plt.subplot(121)
    for i in range(eth, gazebo + 1, 1):
        if i in result_dict:
            plt.plot(x_sim_trans, translation_model_dict[i], label=labels[i])
    ax1 = plt.gca()
    plt.grid(True, 'both', 'y')
    plt.xlabel("translation error / m")
    plt.xlim(TRANS_THRESHOLD_MIN, TRANS_THRESHOLD_MAX)
    plt.ylabel("probability")
    plt.ylim(0.0, 1.0)
    y_ticks = np.arange(0.0, 1.0, 0.1)
    ax1.set_yticks(y_ticks)
    plt.yticks()
    plt.legend(loc = 'best')

    plt.subplot(122)
    for i in range(eth, gazebo + 1, 1):
        if i in result_dict:
            plt.plot(x_sim_rotation, rotation_model_dict[i], label=labels[i])
    ax1 = plt.gca()
    plt.grid(True, 'both', 'y')
    plt.xlabel("rotation error / deg")
    plt.xlim(ROT_THRESHOLD_MIN, ROT_THRESHOLD_MAX)
    plt.ylabel("probability")
    plt.ylim(0.0, 1.0)
    y_ticks = np.arange(0.0, 1.0, 0.1)
    ax1.set_yticks(y_ticks)
    plt.yticks()
    plt.legend(loc = 'best')

    plt.suptitle("ETH-" + type + "-Poses")
    plt.show()
    """

# show results with high overlap
def show_overlap_result(type):
    # Get data from the file
    result_dict = {}
    validation_dict = {}
    for i in range(eth, gazebo + 1, 1):
        if os.path.exists(labels[i] + Result_file):
            result_dict[i] = np.loadtxt(labels[i] + Result_file, float, delimiter=',', skiprows=0)

        if os.path.exists(Path[i] + Validation_file[i]):
            tmp = np.loadtxt(Path[i] + Validation_file[i], str, delimiter=',', skiprows=1)
            validation_dict[i] = tmp[::10]
    LIST_LEN = {}

    # save the items with high overlap
    # result_dict[apartment] = result_dict[apartment][validation_dict[apartment][:, 1] == ' easyPoses']
    for i in range(eth, gazebo + 1, 1):
        if i in result_dict:
            if (type == 'high'):
                result_dict[i] = result_dict[i][validation_dict[i][:, 0].astype(float) >= 0.75]
            elif (type == 'medium'):
                result_dict[i] = result_dict[i][validation_dict[i][:, 0].astype(float) < 0.75]
                validation_dict[i] = validation_dict[i][validation_dict[i][:, 0].astype(float) < 0.75]
                # print("the length is ", len(result_dict[i]))
                result_dict[i] = result_dict[i][validation_dict[i][:, 0].astype(float) > 0.50]
                validation_dict[i] = validation_dict[i][validation_dict[i][:, 0].astype(float) > 0.50]
                # print("the length is ", len(result_dict[i]))
            elif (type == 'low'):
                result_dict[i] = result_dict[i][validation_dict[i][:, 0].astype(float) <= 0.50]
            LIST_LEN[i] = len(result_dict[i])

    TRANS_THRESHOLD_MIN = 0.0
    TRANS_THRESHOLD_MAX = 4.0
    TRANS_STEP = 0.001
    TRANS_NUM = int((TRANS_THRESHOLD_MAX + 2 * TRANS_STEP - TRANS_THRESHOLD_MIN) / TRANS_STEP)
    TRANS_POINT_NUM = TRANS_NUM * 2
    translation_error_cumulative_arr = {}
    for i in range(eth, gazebo + 1, 1):
        if i in result_dict:
            TRANS_THRESHOLD_MIN = 0.0
            TRANS_THRESHOLD_MAX = max(result_dict[i][:, 1])
            TRANS_STEP = 0.01
            TRANS_NUM = int((TRANS_THRESHOLD_MAX + 2 * TRANS_STEP - TRANS_THRESHOLD_MIN) / TRANS_STEP)
            TRANS_POINT_NUM = TRANS_NUM * 2
            translation_error_cumulative_arr[i] = []
            for j in range(TRANS_NUM):
                current_threshold = float(j * TRANS_STEP)
                cur_probability = sum(1 for val in result_dict[i][:, 1] if val <= current_threshold) / LIST_LEN[i]
                translation_error_cumulative_arr[i].append([current_threshold, cur_probability])
            translation_error_cumulative_arr[i] = np.array(translation_error_cumulative_arr[i]) # change it to array
            if not os.path.exists("overlap_" + type):
                os.mkdir("overlap_" + type)
            np.savetxt("overlap_" + type + '/' + labels[i] + "_cumulative_translation_probability", translation_error_cumulative_arr[i], fmt="%.5f", delimiter=',')


    # Create the model dict to store points
    translation_model_dict = {}
    x_sim_trans = np.linspace(TRANS_THRESHOLD_MIN, TRANS_THRESHOLD_MAX, TRANS_POINT_NUM)
    for i in range(eth, gazebo + 1, 1):
        if i in translation_error_cumulative_arr:
            trans_model = make_interp_spline(translation_error_cumulative_arr[i][:, 0], translation_error_cumulative_arr[i][:, 1])
            y_sim_trans = trans_model(x_sim_trans)
            translation_model_dict[i] = y_sim_trans

    ROT_THRESHOLD_MIN = 0.0
    ROT_THRESHOLD_MAX = 25.0
    ROT_STEP = 0.001
    ROT_NUM = int((ROT_THRESHOLD_MAX + 2 * ROT_STEP - ROT_THRESHOLD_MIN) / ROT_STEP)
    ROT_POINT_NUM = ROT_NUM * 2
    # Convert the angles to degrees
    rotation_error_deg_dict = {}
    for i in range(eth, gazebo + 1, 1):
        if i in result_dict:
            ROT_THRESHOLD_MIN = 0.0
            ROT_THRESHOLD_MAX = max(result_dict[i][:, 2]) / math.pi * 180
            ROT_STEP = 0.01
            ROT_NUM = int((ROT_THRESHOLD_MAX + 2 * ROT_STEP - ROT_THRESHOLD_MIN) / ROT_STEP)
            ROT_POINT_NUM = ROT_NUM * 2
            rotation_error_deg_dict[i] = result_dict[i][:, 2] / math.pi * 180

    rotation_error_cumulative_arr = {}
    for i in range(eth, gazebo + 1, 1):
        if i in result_dict:
            rotation_error_cumulative_arr[i] = []
            for j in range(ROT_NUM):
                current_threshold = float(j * ROT_STEP)
                cur_probability = sum(1 for val in rotation_error_deg_dict[i] if val <= current_threshold) / LIST_LEN[i]
                rotation_error_cumulative_arr[i].append([current_threshold, cur_probability])
            rotation_error_cumulative_arr[i] = np.array(rotation_error_cumulative_arr[i])
            if not os.path.exists("overlap_" + type):
                os.mkdir("overlap_" + type)
            np.savetxt("overlap_" + type + '/' + labels[i] + "_cumulative_rotation_probability", rotation_error_cumulative_arr[i], fmt="%.5f", delimiter=',')
    # print(rotation_error_cumulative_arr[apartment])
    rotation_model_dict = {}
    x_sim_rotation = np.linspace(ROT_THRESHOLD_MIN, ROT_THRESHOLD_MAX, ROT_POINT_NUM)
    for i in range(eth, gazebo + 1, 1):
        if i in rotation_error_cumulative_arr:
            trans_model = make_interp_spline(rotation_error_cumulative_arr[i][:, 0], rotation_error_cumulative_arr[i][:, 1])
            y_sim_rotation = trans_model(x_sim_rotation)
            rotation_model_dict[i] = y_sim_rotation

# show_overall_result()

# 'easy', 'medium', 'hard'
# show_pose_result('easy')
# show_pose_result('medium')
# show_pose_result('hard')

# 'low', 'medium', 'high'
show_overlap_result('low')
show_overlap_result('medium')
show_overlap_result('high')
# test()

# print time
# computeConsumedTime()