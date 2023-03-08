# related to DSP
import math
import json
import random
from Signal_processing_utils import sparse_key_smoothing
import librosa
import parselmouth
import sys
sys.path.insert(0, "C:/Users/evansamaa/Documents/GitHub/EvansToolBox/Utils")
from Speech_Data_util import Sentence_word_phone_parser


class SplineInterval:
    def __init__(self, id, min_t, max_t, points, priority=0):
        self.min_t = min_t
        self.max_t = max_t
        self.points = sorted(points)
        self.updated_points = sorted(points)
        self.id = id
        self.priority = priority
    def add_point(self, point_t, updated=False):
        if updated:
            if point_t < self.points[0]:
                self.points.insert(0, point_t)
                return
            elif point_t > self.points[-1]:
                self.points.append(point_t)
                return
            for i in range(1, len(self.points)):
                if point_t < self.points[i]:
                    self.points.insert(i, point_t)
                    return
        else:
            if point_t < self.updated_points[0]:
                self.updated_points.insert(0, point_t)
                return
            elif point_t > self.updated_points[-1]:
                self.updated_points.append(point_t)
                return
            for i in range(1, len(self.updated_points)):
                if point_t < self.updated_points[i]:
                    self.updated_points.insert(i, point_t)
                    return
    def intersect(self, other):
        if self.min_t <= other.max_t and other.min_t <= self.max_t:
            return True
        else:
            return False
    def __contains__(self, t):
        if t >= self.min_t and t <= self.max_t:
            return True
        else:
            return False
class ControlPoints:
    def __init__(self, t, x, owner):
        self.t = t
        self.x = [x]
        self.owners = [owner]
        self.intersected = []
    def get_x(self, id):
        if len(self.owners) == 1:
            return self.x[0]
        else:
            for i in range(0, len(self.owners)):
                if self.owners[i] == id:
                    return self.x[i]
    def sum_x(self):
        out = 0
        for i in range(0, len(self.x)):
            out += self.x[i]
        return out
    def mean_x(self):
        out = 0
        for i in range(0, len(self.x)):
            out += self.x[i]
        return out / len(self.x)
    def update_x(self, new_x):
        self.x = new_x
    def update_x(self, new_t):
        self.t = new_t
    def add_owner(self, new_owner, new_x):
        self.owners.append(new_owner)
        self.x.append(new_x)
    def remove_owner(self, owner, x):
        self.owners.remove(owner)
        self.owners.remove(x)
    def add_intersected(self, new_intersected):
        self.intersected.append(new_intersected)
    def remove_intersected(self, intersected):
        self.intersected.remove(intersected)
    def __str__(self):
        out = "===========\nt = {}\n".format(self.t)
        out_x = "x and its owners are = ["
        for i in range(0, len(self.x)):
            out_x += str(self.x[i]) + " --> " + str(self.owners[i]) + ", "
        out_intersect = "]\nintersect with = ["
        for i in range(0, len(self.intersected)):
            out_intersect += str(self.intersected[i]) + " ,"
        return out + out_x + out_intersect + "]"
class ImpluseSpineCurveModel:
    def __init__(self, base=None):
        if base is None:
            self.control_points = {}
            self.impulse_intervals = []
            self.interval_count = 0
            self.control_pts_refined = []
            self.cached_xt = []
            self.cached_x = []
    def _interpolate(self, arr_t, t, interval_id, print_stff = False):
        if t <= arr_t[0]:
            return self.control_points[arr_t[0]].get_x(interval_id)
        if t >= arr_t[-1]:
            return self.control_points[arr_t[-1]].get_x(interval_id)
        for i in range(0, len(arr_t) - 1):
            if arr_t[i] <= t and arr_t[i + 1] > t:
                interval_index = i
                break
        x0 = self.control_points[arr_t[interval_index]].get_x(interval_id)
        x1 = self.control_points[arr_t[interval_index+1]].get_x(interval_id)
        rtv = x0 + (t - arr_t[interval_index]) / (arr_t[interval_index + 1] - arr_t[interval_index]) * (x1 - x0)
        if print_stff:
            print(t, x0, x1, arr_t[interval_index], arr_t[interval_index+1])

        return rtv
    def add_to_interval_tier(self, points, priority):
        # points are expected to be [[time] [space]], each of shape [n,]
        # transition_interval is be of shape [2, ] such that the transition interval is within
        # time[x], time[y]
        t, x = points
        this_interval = []
        min_t = 100000
        max_t = -100000
        for i in range(0, len(t)):
            min_t = min(min_t, t[i])
            max_t = max(max_t, t[i])
            try:
                # add the interval as an owner to an existing point
                existant_ctpt:ControlPoints = self.control_points[t[i]]
                existant_ctpt.add_owner(self.interval_count, x[i])
                this_interval.append(t[i])
            except:
                # add a new control point to the interval
                new_ctpt = ControlPoints(t[i], x[i], self.interval_count)
                this_interval.append(t[i])
                self.control_points[t[i]] = new_ctpt
        new_interval = SplineInterval(self.interval_count, min_t, max_t, this_interval, priority)
        self.impulse_intervals.append(new_interval)
        # update this index
        self.interval_count += 1
        return
    def fill_holes(self):
        # call this to create intervals that fill holes in the list of intervals, so that the base additive interval
        # always have a non-zero value, and other tiers can be added to this
        sorted_interval_list = sorted(self.impulse_intervals, key=lambda x: x.min_t)
        for i in range(0, len(sorted_interval_list)-1):
            current_interval = sorted_interval_list[i]
            next_interval = sorted_interval_list[i + 1]
            if current_interval.max_t >= next_interval.min_t:
                pass
            else:
                start_v = self.control_points[current_interval.max_t].mean_x()
                end_v = self.control_points[next_interval.min_t].mean_x()
                self.add_to_interval_tier([[current_interval.max_t, next_interval.min_t], [start_v, end_v]], 0)
    def recompute_all_points(self, mean=False):
        sorted_interval_list = sorted(self.impulse_intervals, key=lambda x: x.max_t)
        # for each point, check if they exist in any interval that does not own it
        for pt_t in self.control_points:
            added = False
            for i in range(self.interval_count):
                interval = sorted_interval_list[i]
                if pt_t in interval:
                    added = True
                    if (not interval.id in self.control_points[pt_t].owners
                            and not interval.id in self.control_points[pt_t].intersected):
                        self.control_points[pt_t].add_intersected(interval.id)
                        self.impulse_intervals[interval.id].add_point(pt_t)
                elif pt_t > interval.max_t and added:
                    break
                else:
                    continue
        # if the intersection handling is to sum the signal
        self.control_pts_refined = []
        refined_t = []
        refined_x = []
        # sum up the control points
        for pt_t in self.control_points:
            pt:ControlPoints = self.control_points[pt_t]
            additive_tier_value = 0
            additive_tier_count = 0
            mean_tier_value = 0
            mean_tier_count = 0
            override_tier_value = 0
            override_tier_count = 0
            override_tier_owner_count = 0
            # obtaining value of each owner and intersection:
            for owner in pt.owners:
                priority = self.impulse_intervals[owner].priority
                if priority == 0:
                    additive_tier_value += pt.get_x(owner)
                    additive_tier_count += 1
                elif priority == 5:
                    # if there are multiple priority 5 intervals, we calculate a mean
                    override_tier_value += pt.get_x(owner)
                    override_tier_count += 1
                    override_tier_owner_count += 1
                else:
                    mean_tier_value += pt.get_x(owner)
                    mean_tier_count += 1
            # if interval_ignored:
            #     continue
            for intersected_i in range(len(pt.intersected)):
                priority = self.impulse_intervals[pt.intersected[intersected_i]].priority
                if priority == 0:
                    additive_tier_value += self._interpolate(self.impulse_intervals[pt.intersected[intersected_i]].points, pt_t, pt.intersected[intersected_i])
                    additive_tier_count += 1
                elif priority == 5:
                    # if there are multiple priority 5 intervals, we calculate a mean
                    override_tier_value += self._interpolate(self.impulse_intervals[pt.intersected[intersected_i]].points, pt_t, pt.intersected[intersected_i])
                    override_tier_count += 1
                else:
                    mean_tier_value += self._interpolate(self.impulse_intervals[pt.intersected[intersected_i]].points, pt_t, pt.intersected[intersected_i])
                    mean_tier_count += 1
            # compute mean for each type of value
            if additive_tier_count > 0:
                additive_tier_value = additive_tier_value / additive_tier_count
            if mean_tier_count > 0:
                mean_tier_value = mean_tier_value / mean_tier_count
            if override_tier_count > 0:
                override_tier_value = override_tier_value / override_tier_count

            # if there is an override tier, ignore all other tiers
            if override_tier_count > 0 and override_tier_owner_count == 0:
                continue
            elif override_tier_count > 0:
                refined_x.append(override_tier_value)
                refined_t.append(pt_t)
            else:
                refined_x.append(mean_tier_value + additive_tier_value)
                refined_t.append(pt_t)
        # sort the two lists
        refined_x = [x for _, x in sorted(zip(refined_t, refined_x), key=lambda pair: pair[0])]
        refined_t = sorted(refined_t)

        # cache the result and output
        self.cached_x = refined_x
        self.cached_xt = refined_t
        return [refined_t, refined_x]

    def sum_with_other_tiers(self, others):
        # this sums
        new_tier = ImpluseSpineCurveModel()
        others.append(self)
        for tier in others:
            for interval in tier.impulse_intervals:
                point_set_x = []
                point_set_t = []
                for pt in interval.points:
                    point_set_x.append(tier.control_points[pt].get_x(interval.id))
                    point_set_t.append(pt)
                new_tier.add_to_interval_tier([point_set_t, point_set_x], interval.priority)
        return new_tier.recompute_all_points()

def np_sum(arr):
    out = 0
    for i in arr:
        out += i
    return out
def np_mean_and_std(arr):
    mean = np_sum(arr)/len(arr)
    std = 0
    for i in arr:
        std += (i - mean)**2
    std = std / (len(arr)-1)
    std = math.sqrt(std)
    return mean, std
def np_max_min(arr):
    max_val = arr[0]
    min_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] >= max_val:
            max_val = arr[i]
        if arr[i] <= min_val:
            min_val = arr[i]
    return max_val, min_val
def np_sum_two(arr1, arr2):
    out = []
    for i in range(0, len(arr2)):
        out.append(arr1[i] + arr2[i])
    return out
def np_minus(arr1, arr2):
    out = []
    for i in range(0, len(arr2)):
        out.append(arr1[i] - arr2[i])
    return out
def np_argmax(arr):
    max_val = arr[0]
    conter = 0
    for i in range(0, len(arr)):
        if arr[i] > max_val:
            conter = i
            max_val = arr[i]
    return conter
def np_softmax(arr):
    if len(arr) == 0:
        return 1
    denom = 0
    for i in range(0, len(arr)):
        arr[i] = math.exp(arr[i])
        denom += arr[i]
    for i in range(0, len(arr)):
        arr[i] /= denom
    return arr[i]
class signal_stats:
    def __init__(self, intensity, pitch):
        if len(intensity) != 0:
            self.excursion_pitch = 0
            pitch_dt_1 = pitch[1:]
            pitch_prime = np_minus(pitch[0:-1], pitch_dt_1)
            self.excursion_f = abs(np_sum(pitch_prime))
            self.mean_i, self.std_i = np_mean_and_std(intensity)
            self.mean_f, self.std_f = np_mean_and_std(pitch)
            self.max_i, self.min_i = np_max_min(intensity)
            self.max_f, self.min_f = np_max_min(pitch)
        else:
            self.excursion_f = 0
            self.mean_i = 0
            self.std_i = 0
            self.mean_f = 0
            self.std_f = 0
            self.min_i = 0
            self.max_i = 0
            self.min_f = 0
            self.max_f = 0
def laplancian_smoothing(data, iterations=1):
    for j in range(0, iterations):
        for i in range(1, data.shape[0] - 1):
            data[i] = data[i - 1:i + 2].mean()
    return data
def get_intensity(x):
    total_time_steps = math.ceil(len(x) / 441)
    kaiser_window = [0.000356294, 0.000434478, 0.000520568, 0.000615029, 0.000718343, 0.000831007, 0.000953531, 0.00108645,
         0.00123029, 0.00138563, 0.00155304, 0.0017331, 0.00192643, 0.00213364, 0.00235538, 0.0025923, 0.00284507,
         0.00311438, 0.00340092, 0.00370541, 0.00402859, 0.0043712, 0.00473401, 0.00511779, 0.00552334, 0.00595146,
         0.00640297, 0.0068787, 0.00737952, 0.00790627, 0.00845984, 0.00904112, 0.009651, 0.0102904, 0.0109602,
         0.0116615, 0.012395, 0.0131618, 0.0139629, 0.0147993, 0.0156718, 0.0165815, 0.0175295, 0.0185167, 0.0195442,
         0.0206129, 0.0217239, 0.0228783, 0.024077, 0.0253211, 0.0266117, 0.0279498, 0.0293365, 0.0307728, 0.0322598,
         0.0337984, 0.0353898, 0.037035, 0.0387351, 0.0404911, 0.0423039, 0.0441748, 0.0461045, 0.0480944, 0.0501451,
         0.0522579, 0.0544336, 0.0566733, 0.0589778, 0.0613483, 0.0637855, 0.0662906, 0.0688641, 0.0715072, 0.0742207,
         0.0770055, 0.0798622, 0.082792, 0.0857952, 0.088873, 0.092026, 0.0952549, 0.0985603, 0.101943, 0.105404,
         0.108943, 0.112561, 0.116259, 0.120037, 0.123895, 0.127835, 0.131856, 0.13596, 0.140145, 0.144413, 0.148764,
         0.153198, 0.157715, 0.162315, 0.166999, 0.171767, 0.176619, 0.181554, 0.186572, 0.191674, 0.19686, 0.202128,
         0.20748, 0.212914, 0.218431, 0.224029, 0.229709, 0.23547, 0.241311, 0.247232, 0.253233, 0.259312, 0.26547,
         0.271704, 0.278015, 0.284401, 0.290861, 0.297396, 0.304002, 0.31068, 0.317429, 0.324246, 0.331131, 0.338083,
         0.3451, 0.352181, 0.359324, 0.366529, 0.373792, 0.381113, 0.388491, 0.395923, 0.403407, 0.410943, 0.418527,
         0.426159, 0.433836, 0.441557, 0.449318, 0.457119, 0.464957, 0.47283, 0.480735, 0.488672, 0.496636, 0.504626,
         0.512641, 0.520675, 0.52873, 0.5368, 0.544884, 0.55298, 0.561083, 0.569194, 0.577308, 0.585423, 0.593536,
         0.601645, 0.609746, 0.617838, 0.625917, 0.633981, 0.642026, 0.650051, 0.658051, 0.666024, 0.673969, 0.68188,
         0.689757, 0.697595, 0.705392, 0.713145, 0.720852, 0.728509, 0.736112, 0.743661, 0.751151, 0.75858, 0.765944,
         0.773242, 0.780469, 0.787625, 0.794705, 0.801706, 0.808626, 0.815462, 0.822213, 0.828873, 0.835441, 0.841916,
         0.848292, 0.854568, 0.860742, 0.866811, 0.872773, 0.878625, 0.884364, 0.889988, 0.895494, 0.900881, 0.906147,
         0.911288, 0.916303, 0.92119, 0.925946, 0.93057, 0.935058, 0.939412, 0.943627, 0.947701, 0.951633, 0.955423,
         0.959067, 0.962564, 0.965913, 0.969111, 0.97216, 0.975055, 0.977796, 0.980384, 0.982814, 0.985087, 0.987203,
         0.989158, 0.990955, 0.99259, 0.994064, 0.995376, 0.996525, 0.99751, 0.998333, 0.998991, 0.999486, 0.999814,
         0.999979, 0.999979, 0.999814, 0.999486, 0.998991, 0.998333, 0.99751, 0.996525, 0.995376, 0.994064, 0.99259,
         0.990955, 0.989158, 0.987203, 0.985087, 0.982814, 0.980384, 0.977796, 0.975055, 0.97216, 0.969111, 0.965913,
         0.962564, 0.959067, 0.955423, 0.951633, 0.947701, 0.943627, 0.939412, 0.935058, 0.93057, 0.925946, 0.92119,
         0.916303, 0.911288, 0.906147, 0.900881, 0.895494, 0.889988, 0.884364, 0.878625, 0.872773, 0.866811, 0.860742,
         0.854568, 0.848292, 0.841916, 0.835441, 0.828873, 0.822213, 0.815462, 0.808626, 0.801706, 0.794705, 0.787625,
         0.780469, 0.773242, 0.765944, 0.75858, 0.751151, 0.743661, 0.736112, 0.728509, 0.720852, 0.713145, 0.705392,
         0.697595, 0.689757, 0.68188, 0.673969, 0.666024, 0.658051, 0.650051, 0.642026, 0.633981, 0.625917, 0.617838,
         0.609746, 0.601645, 0.593536, 0.585423, 0.577308, 0.569194, 0.561083, 0.55298, 0.544884, 0.5368, 0.52873,
         0.520675, 0.512641, 0.504626, 0.496636, 0.488672, 0.480735, 0.47283, 0.464957, 0.457119, 0.449318, 0.441557,
         0.433836, 0.426159, 0.418527, 0.410943, 0.403407, 0.395923, 0.388491, 0.381113, 0.373792, 0.366529, 0.359324,
         0.352181, 0.3451, 0.338083, 0.331131, 0.324246, 0.317429, 0.31068, 0.304002, 0.297396, 0.290861, 0.284401,
         0.278015, 0.271704, 0.26547, 0.259312, 0.253233, 0.247232, 0.241311, 0.23547, 0.229709, 0.224029, 0.218431,
         0.212914, 0.20748, 0.202128, 0.19686, 0.191674, 0.186572, 0.181554, 0.176619, 0.171767, 0.166999, 0.162315,
         0.157715, 0.153198, 0.148764, 0.144413, 0.140145, 0.13596, 0.131856, 0.127835, 0.123895, 0.120037, 0.116259,
         0.112561, 0.108943, 0.105404, 0.101943, 0.0985603, 0.0952549, 0.092026, 0.088873, 0.0857952, 0.082792,
         0.0798622, 0.0770055, 0.0742207, 0.0715072, 0.0688641, 0.0662906, 0.0637855, 0.0613483, 0.0589778, 0.0566733,
         0.0544336, 0.0522579, 0.0501451, 0.0480944, 0.0461045, 0.0441748, 0.0423039, 0.0404911, 0.0387351, 0.037035,
         0.0353898, 0.0337984, 0.0322598, 0.0307728, 0.0293365, 0.0279498, 0.0266117, 0.0253211, 0.024077, 0.0228783,
         0.0217239, 0.0206129, 0.0195442, 0.0185167, 0.0175295, 0.0165815, 0.0156718, 0.0147993, 0.0139629, 0.0131618,
         0.012395, 0.0116615, 0.0109602, 0.0102904, 0.009651, 0.00904112, 0.00845984, 0.00790627, 0.00737952, 0.0068787,
         0.00640297, 0.00595146, 0.00552334, 0.00511779, 0.00473401, 0.0043712, 0.00402859, 0.00370541, 0.00340092,
         0.00311438, 0.00284507, 0.0025923, 0.00235538, 0.00213364, 0.00192643, 0.0017331, 0.00155304, 0.00138563,
         0.00123029, 0.00108645, 0.000953531, 0.000831007, 0.000718343, 0.000615029, 0.000520568, 0.000434478,
         0.000356294]
    kaiser_sum = np_sum(kaiser_window)
    intensity = []
    for i in range(0, int(total_time_steps)):
        end_up = min((i + 1) * 441, len(x))
        current_kaiser_window = kaiser_window[0:(end_up - i * 441)]
        current = current_kaiser_window * x[i * 441:end_up] * x[i * 441:end_up]
        current_frame_intensity = np_sum(current) / kaiser_sum / 4.0e-10
        # current_frame_intensity = current.sum()/kaiser_window.sum()
        if current_frame_intensity <= 1.0e-30:
            current_frame_intensity = 0
        else:
            # current_frame_intensity = np.sqrt(current_frame_intensity)
            current_frame_intensity = 10 * math.log10(current_frame_intensity)
        intensity.append(current_frame_intensity)
    return intensity
def load_praatoutput(file_name):
    phone_prosodic_list = []
    phone_list = []
    phone_intervals = []
    word_list = []
    word_intervals = []
    stats = []
    f = open(file_name)
    garb = f.readline()
    arr = f.readlines()
    for i in range(0, len(arr)):
        content = arr[i].split("\t")
        start = float(content[0])
        end = float(content[1])
        phone = content[21]
        word = content[26]
        phone_list.append(phone)
        word_list.append(word)
        phone_intervals.append([start, end])
    #     merged_word_list = []
    merged_word_intervals = []

    prev_word = word_list[0]
    merged_word_intervals.append([phone_intervals[0][0]])

    prev_k = 1
    for i in range(1, len(word_list)):
        if word_list[i] != prev_word:
            merged_word_intervals[-1].append(phone_intervals[i - 1][1])
            for j in range(0, prev_k - 1):
                merged_word_intervals.append(merged_word_intervals[-1])
            prev_word = word_list[i]
            merged_word_intervals.append([phone_intervals[i][0]])
            prev_k = 0
        if i == len(word_list) - 1:
            merged_word_intervals[-1].append(phone_intervals[i - 1][1])
            for j in range(0, prev_k):
                merged_word_intervals.append(merged_word_intervals[-1])
        prev_k = prev_k + 1
    return phone_list, phone_intervals, word_list, merged_word_intervals

# related to parsing
class XSampa_phonemes_dicts():
    def strip(self, phone):
        try:
            float(phone[-1])
            return phone[:-1]
        except:
            return phone

    def __init__(self):
        self.vocabs = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
                           'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH',
                           'UH',
                           'UW', 'V', 'W', 'Y', 'Z', 'ZH', "sil", "sp"])
        self.vowels = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
                           'IH', 'IY', 'OW', 'OY', 'UH', 'UW', ])
        self.voiced = set(['M', 'N', "L", "NG"]).union(self.vowels)
        self.consonants = set(['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG',
                               'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'])
        self.consonants_no_jaw = self.consonants
        self.lip_closer = set(["B", "F", "M", "P", "S", "V"])
        self.lip_rounder = set(["B", "F", "M", "P", "V"])
        self.nasal_obtruents = set(['L', 'N', 'NG', 'T', 'D', 'G', 'K', 'F', 'V', 'M', 'B', 'P'])
        self.fricative = set(["S", "Z", "ZH", "SH", "CH", "F", "V", 'TH'])
        self.plosive = set(["P", "B", "D", "T", "K", "G"])
        self.lip_heavy = set(["W", "OW", "UW", "S", "Z", "Y", "JH", "OY"])
        self.sibilant = set(["S", "Z", "SH", "CH", "ZH"])
class Check_Pauses():
    def __init__(self):
        self.filled_pauses = set(["um", "ah", "uh", "eh"])

    def __call__(self, word):
        if word == ".":
            return "SP"  # pause
        if word[1:].lower() in self.filled_pauses:
            return "FP"  # filled pause
        else:
            return "NP"  # no pause
class Sentence_word_phone_pointer_structure:
    def __init__(self, praat_file_path, text_file_path):
        phone_list, phone_intervals, word_list, word_intervals = load_praatoutput(praat_file_path)
        sentence_list, sentence_intervals = phone_to_sentences(text_file_path,
                                                               phone_list, phone_intervals, word_list, word_intervals)
        self.phone_list = phone_list
        self.phone_intervals = phone_intervals
        self.phone_to_word = word_list
        self.phone_to_sentence = sentence_list

        self.word_list = []
        self.word_intervals = []
        self.word_to_phone = []
        self.word_to_sentence = []

        self.sentence_list = []
        self.sentence_intervals = []
        self.sentence_to_phone = []
        self.sentence_to_word = []

        # get all word level pointers
        start = 0
        for i in range(1, len(word_list)):
            if word_list[i - 1] != word_list[i]:
                self.word_list.append(word_list[i - 1])
                self.word_intervals.append(word_intervals[i - 1])
                word_to_phone_temp = []
                for j in range(start, i):
                    word_to_phone_temp.append(j)
                start = i
                self.word_to_phone.append(word_to_phone_temp)
                self.word_to_sentence.append(self.phone_to_sentence[i - 1])

            if i == len(word_list) - 1:

                self.word_list.append(word_list[i])
                self.word_intervals.append(word_intervals[i])
                word_to_phone_temp = []
                for j in range(start, i + 1):
                    word_to_phone_temp.append(j)
                start = i
                self.word_to_phone.append(word_to_phone_temp)
                self.word_to_sentence.append(self.phone_to_sentence[i])
        # get all sentence level pointers
        self.sentence_list = []
        for i in range(0, self.phone_to_sentence[-1] + 1):
            self.sentence_list.append(i)
            for j in range(0, len(sentence_intervals)):
                if self.phone_to_sentence[j] == i:
                    self.sentence_intervals.append(sentence_intervals[j])
                    break
        start = 0
        for i in range(1, len(self.phone_to_sentence)):
            if self.phone_to_sentence[i - 1] != self.phone_to_sentence[i]:
                sentence_to_phone_temp = []
                for j in range(start, i):
                    sentence_to_phone_temp.append(j)
                start = i
                self.sentence_to_phone.append(sentence_to_phone_temp)

            if i == len(self.phone_to_sentence) - 1:
                sentence_to_phone_temp = []
                for j in range(start, i + 1):
                    sentence_to_phone_temp.append(j)
                start = i
                self.sentence_to_phone.append(sentence_to_phone_temp)

        start = 0
        for i in range(1, len(self.word_to_sentence)):
            if self.word_to_sentence[i - 1] != self.word_to_sentence[i]:
                sentence_to_word_temp = []
                for j in range(start, i):
                    sentence_to_word_temp.append(j)
                start = i
                self.sentence_to_word.append(sentence_to_word_temp)

            if i == len(self.word_to_sentence) - 1:
                sentence_to_word_temp = []
                for j in range(start, i + 1):
                    sentence_to_word_temp.append(j)
                start = i
                self.sentence_to_word.append(sentence_to_word_temp)
def get_pitch(x):
    snd = parselmouth.Sound(x)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    return pitch_values
def outputAsyncToFile(path, output):
    # the input should be in the form of a list of lists [6 of [n, ]]
    # They should be [xt, x, yt, y, zt, z]
    output = {"xt": output[0], "x": output[1],
              "yt": output[2], "y": output[3],
              "zt": output[4], "z": output[5]}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f)
    return
def interpolate_value(arr_t, arr_x, t):
    arr_t = sorted(arr_t)
    arr_x = [x for _, x in sorted(zip(arr_t, arr_x))]

    if t <= arr_t[0]:
        return arr_x[0]
    if t >= arr_t[-1]:
        return arr_x[-1]
    for i in range(0, len(arr_t) - 1):
        if arr_t[i] <= t and arr_t[i + 1] > t:
            interval_index = i
            break
    rtv = arr_x[interval_index] + (t - arr_t[interval_index]) / (arr_t[interval_index + 1] - arr_t[interval_index]
                                                                 ) * (arr_x[interval_index + 1] - arr_x[interval_index])
    return rtv
def phone_to_sentences(text_file_path, phone_list, phone_intervals, word_list, word_intervals):
    # here I will label each phone to be part of a sentence
    punctuations = set([",", ".", "!", "?", ";", "\n", "...", "......", ":", ""])
    pause_checker = Check_Pauses()
    text_file = open(text_file_path, "r").readlines()
    # process the script to add punctuation to it
    all_script = " ".join(text_file)
    raw_script = all_script.split(" ")
    raw_script_copy = []
    for i in range(0, len(raw_script)):
        raw_script[i] = raw_script[i].strip()
        if len(raw_script[i]) == 0:
            continue
        if raw_script[i][0] == "<" and raw_script[i][-1] == ">":
            continue
        if raw_script[i] in punctuations:
            raw_script_copy.append(raw_script[i])
        elif raw_script[i][-1] in punctuations:
            raw_script_copy.append(raw_script[i][:-1])
            raw_script_copy.append(raw_script[i][-1])
        else:
            raw_script_copy.append(raw_script[i])
    raw_script = raw_script_copy
    sentence_number = 0
    sentence_number_tags = []
    sentence_intervals = []

    # two pointers, one for the textfile, the other one for phoneme list
    pt1 = 0
    pt2 = 0
    start = 0
    end = 0
    prev_word = "."
    # first the very first word in the list
    for i in range(len(word_list)):
        if word_list[i] != ".":
            prev_word = word_list[i][1:].lower()
            break

    # iteratively give each phoneme a sentence tag
    added = False
    while pt2 < len(phone_list):
        word_with_punctuation = raw_script[pt1].lower()
        # remove the underscore at the beginning of words
        if len(word_list[pt2]) > 1:
            word_from_praatscript = word_list[pt2].lower()[1:]
        else:
            word_from_praatscript = word_list[pt2].lower()

        if word_from_praatscript != "." and word_from_praatscript != prev_word:
            pt1 += 1
            pt1 = min(pt1, len(raw_script) - 1)
            word_with_punctuation = raw_script[pt1].lower()
            if word_with_punctuation in punctuations:
                pt1 += 1
                pt1 = min(pt1, len(raw_script) - 1)
                prev_word = raw_script[pt1].lower()
                if not added:
                    sentence_number += 1
            else:
                prev_word = word_from_praatscript
                added = False

        elif word_from_praatscript != "." and word_from_praatscript == prev_word:
            added = False

        # if the current word form the praatscript label is silence, then go to a new sentence
        if word_from_praatscript == ".":
            # if the file starts with a pause, don't do anything
            if sentence_number == 0 and pt2 == 0:
                pass
            elif word_intervals[pt2][1] - word_intervals[pt2][0] > 0.4:
                # otherwise increment the sentence number by 1
                sentence_number += 1
                added = True
        sentence_number_tags.append(sentence_number)
        pt2 += 1
    # using the sentence tag to give each phoneme a sentence interval
    start = word_intervals[0][0]
    current_sentence = 0
    prev_index = 0
    for i in range(0, len(word_list)):
        if sentence_number_tags[i] != current_sentence or i == len(word_list) - 1:
            end = word_intervals[i][0]
            if i == len(word_list) - 1:
                for j in range(prev_index, i + 1):
                    sentence_intervals.append([start, end])
            else:
                for j in range(prev_index, i):
                    sentence_intervals.append([start, end])
            start = end
            prev_index = i
            current_sentence += 1

    return sentence_number_tags, sentence_intervals
def get_praat_intensity_or_pitch(path):
    out = []
    with open(path, "r") as file:
        lines = file.readlines()
        for i in range(1, len(lines)):
            time, val = lines[i].split(",")
            try:
                out.append(float(val))
            except:
                out.append(0)
    return out
def gen_normal(mean, std):
    v1 = random.random()
    v2 = random.random()
    z1 = math.sqrt(-1 * math.log(v1)) * math.cos(2*math.pi*v2)
    z2 = math.sqrt(-1 * math.log(v2)) * math.cos(2*math.pi*v1)
    return z1 * std + mean
class NeckCurve:
    def __init__(self, file_path):
        self.xsampa = XSampa_phonemes_dicts()
        self.out_path = file_path.split(".")[0] + "_neck.json"
        self.input_script_path = file_path.split(".")[0] + ".txt"
        self.input_praatscript = file_path.split(".")[0] + "_PraatOutput.txt"
        self.input_wav_path = file_path.split(".")[0] + ".wav"
        # self.intensity_praat_path = file_path.split(".")[0] + "_intensity.txt"
        # self.pitch_praat_path = file_path.split(".")[0] + "_pitch.txt"
        self.isPause = Check_Pauses()
        self.phone_dict = XSampa_phonemes_dicts()
        self.word_not_emphasized = set(["and", "of", "about", "above", "across", "after", "at", "by", "for", "in", "into",
                                   "off", "on", "since", "through", "until", "with", "into", "onto", "as", "per",
                                   "than"])
        self.negative_affirmation = set(
            ["not", "no", "none", "never", "nah", "negative", "disappointed", "wrong", "incorrect", "negative",
             "won't", "can't", "isn't", "aren't", "didn't", "don't", "doesn't", "shouldn't", "couldn't",
             "wouldn't", "weren't", "wasn't", "mustn't", "needn't", "hadn't", "haven't", "hasn't", "nobody",
             "nothing", "nowhere"])

        # compute intensity and pitch values
        snd = parselmouth.Sound(self.input_wav_path)
        pitch = snd.to_pitch(time_step=0.01)
        pitch_values = pitch.selected_array['frequency']
        self.intensity = get_intensity(snd.values[0])
        pre_pad = int((len(self.intensity) - pitch_values.shape[0]) / 2)
        post_pad = len(self.intensity) - pitch_values.shape[0] - pre_pad
        self.pitch = [0] * pre_pad + pitch_values.tolist() + [0] * post_pad

        # set default parameter values
        self.head_shakes = True
        self.threshold_from_previous_jitter = 0.7
        self.time_between_emphasis = 0.5
        self.head_sentence_level = 2
        self.head_sentence_level_z = 2
        self.base_number_of_shakes = 8
        self.emphasis_max_rotation = 2
        self.max_ambient_movement = 1
        self.left_side_bias = 0.1
        self.random_seed = 0
        self.compute_curve()
    def compute_curve(self):

        # compute tunable values
        self.semantics_script = Sentence_word_phone_parser(
            self.input_praatscript,
            self.input_script_path,
        )

        # obtaining phoneme_level stats:
        self.phone_stats = []
        for i in range(0, len(self.semantics_script.phone_list)):
            interval = self.semantics_script.phone_intervals[i]
            phone = self.semantics_script.phone_list[i]
            phone_intensity = self.intensity[int(interval[0] * 100):int(interval[1] * 100)]
            phone_pitch = self.pitch[int(interval[0] * 100):min(int(interval[1] * 100), len(self.pitch))]
            if self.xsampa.strip(phone) in self.xsampa.vowels:
                current_phone_stats = signal_stats(phone_intensity, phone_pitch)
            else:
                current_phone_stats = signal_stats(phone_intensity, [0, 0])
            self.phone_stats.append(current_phone_stats)

            # obtaining word_level stats:
        self.word_stats = []
        for i in range(0, len(self.semantics_script.word_list)):
            interval = self.semantics_script.word_intervals[i]
            word = self.semantics_script.word_list[i]
            word_pitch = self.pitch[int(interval[0] * 100):min(int(interval[1] * 100), len(self.pitch))]
            word_intensity = self.intensity[int(interval[0] * 100):int(interval[1] * 100)]
            current_word_stats = signal_stats(word_intensity, word_pitch)
            self.word_stats.append(current_word_stats)

            # obtainging sentence_level_stats:
        self.sentence_stats = []
        for i in range(0, len(self.semantics_script.sentence_list)):
            interval = self.semantics_script.sentence_intervals[i]
            sentence = self.semantics_script.sentence_list[i]
            sentence_intensity = self.intensity[int(interval[0] * 100):int(interval[1] * 100)]
            sentence_pitch = self.pitch[int(interval[0] * 100):min(int(interval[1] * 100), len(self.pitch))]
            current_sentence_stats = signal_stats(sentence_intensity, sentence_pitch)
            self.sentence_stats.append(current_sentence_stats)
        self.global_stats = signal_stats(self.intensity, self.pitch)

        impulse_curve_set = ImpluseSpineCurveModel()
        jitter_curve_set = ImpluseSpineCurveModel()
        sentence_curve_set = ImpluseSpineCurveModel()

        impulse_curve_set_y = ImpluseSpineCurveModel()
        sentence_curve_set_z = ImpluseSpineCurveModel()

        # keep track of sentence level for x
        current_level = 0
        prev_level = 0
        # keep track of sentence level for z
        current_level_z = 0
        prev_level_z = 0

        prev_big_spike_time = 0

        # remember the words that are previously emphasized
        emphasized_words = []
        # remember which sentences are negative affirmations
        negative_sentences = []
        
        
        print(len(self.semantics_script.sentence_to_word))
        for i in range(0, len(self.semantics_script.sentence_to_word)):
            sentence_level_transience_head_levels_xt = []
            sentence_level_transience_head_levels_x = []
            sentence_level_stable_head_levels_xt = []
            sentence_level_stable_head_levels_x = []
            #     sentence_level_transience_head_levels_zt = []
            #     sentence_level_transience_head_levels_z = []
            sentence_level_head_levels_z = []
            sentence_level_head_levels_zt = []

            # get the list of words in the transcript
            interval = self.semantics_script.sentence_to_word[i]

            # check to see if the sentence is negative
            negative_word_id = []
            if self.head_shakes:
                for word_id in range(len(interval)):
                    word = self.semantics_script.word_list[interval[word_id]]
                    if self.semantics_script.word_list[interval[word_id]] != "." and self.semantics_script.word_list[
                                                                                    interval[word_id]][
                                                                                1:] in self.negative_affirmation:
                        negative_word_id.append(word_id)
                if len(negative_word_id) > 0:
                    negative_sentences.append(True)
                else:
                    negative_sentences.append(False)
            else:
                negative_sentences.append(False)

            if self.semantics_script.sentence_intervals[i][1] - self.semantics_script.sentence_intervals[i][0] >= 0.4:
                # ====================================== add sentence interval ======================================
                # place the first key frame before the sentence has started
                # make sure that this frame is not silence
                starting_word_id = 0
                for l in range(0, len(interval)):
                    if self.semantics_script.word_list[interval[l]] != ".":
                        starting_word_id = l
                        break
                sentence_level_transience_head_levels_xt.append(
                    self.semantics_script.word_intervals[interval[starting_word_id]][0] - 0.2)
                sentence_level_transience_head_levels_x.append(prev_level * 0.7)

                # the second key will bring the head to certain level
                relative_sentence_intensity = (self.sentence_stats[i].mean_f - self.global_stats.mean_f)
                if len(negative_word_id) > 0:
                    current_level = prev_level * 0.9
                if relative_sentence_intensity >= 0:
                    current_level = gen_normal(self.head_sentence_level / 2, self.head_sentence_level / 4)
                    if abs(relative_sentence_intensity) > self.global_stats.std_f:
                        current_level = gen_normal(self.head_sentence_level, self.head_sentence_level / 4)
                else:
                    current_level = gen_normal(-self.head_sentence_level / 2, self.head_sentence_level / 4)
                    if abs(relative_sentence_intensity) > self.global_stats.std_f:
                        current_level = gen_normal(-self.head_sentence_level, self.head_sentence_level / 4)

                        # add the level to sustain at, this is slower when the distance to travel is great
                delta_head_level_x = abs(current_level - prev_level)
                if delta_head_level_x >= 6:
                    sentence_level_transience_head_levels_xt.append(sentence_level_transience_head_levels_xt[-1] + 0.45)
                    sentence_level_stable_head_levels_xt.append(sentence_level_transience_head_levels_xt[-2] + 0.45)
                else:
                    sentence_level_transience_head_levels_xt.append(sentence_level_transience_head_levels_xt[-1] + 0.35)
                    sentence_level_stable_head_levels_xt.append(sentence_level_transience_head_levels_xt[-2] + 0.35)
                sentence_level_transience_head_levels_x.append(current_level)
                sentence_level_stable_head_levels_x.append(current_level)

                # add additional key to give the motion the sharkfin characteristic
                shark_fin_t = (sentence_level_transience_head_levels_xt[1] -
                               sentence_level_transience_head_levels_xt[0]) * 0.54 + \
                              sentence_level_transience_head_levels_xt[0]
                shark_fin_x = (sentence_level_transience_head_levels_x[1] -
                               sentence_level_transience_head_levels_x[0]) * 0.8 + \
                              sentence_level_transience_head_levels_x[0]
                sentence_level_transience_head_levels_x.insert(1, shark_fin_x)
                sentence_level_transience_head_levels_xt.insert(1, shark_fin_t)

                # add the end point(do not add if the interval is too short)
                end_point_t = self.semantics_script.word_intervals[interval[-1]][1] - 0.4
                if end_point_t > sentence_level_transience_head_levels_xt[-1]:
                    sentence_level_stable_head_levels_xt.append(self.semantics_script.word_intervals[interval[-1]][1] - 0.4)
                    sentence_level_stable_head_levels_x.append(current_level * 0.8)

                # if it's the final sentence, extend the curve into the future so the level won't drop suddenly
                if i >= len(self.semantics_script.sentence_to_word) - 1:
                    sentence_level_stable_head_levels_xt.append(self.semantics_script.word_intervals[interval[-1]][1] + 2)
                    sentence_level_stable_head_levels_x.append(current_level * 0.8)
                prev_level = current_level
                # =============== sentence level head tilt when the intensity is lower than average ===============:
                z_delay = -0.2
                sentence_level_head_levels_zt.append(
                    self.semantics_script.word_intervals[interval[starting_word_id]][0] - 0.2 + z_delay)
                sentence_level_head_levels_z.append(prev_level_z * 0.7)
                if self.sentence_stats[i].mean_i < self.global_stats.mean_i - 0.5 * self.global_stats.std_i and abs(
                        prev_level_z) >= self.head_sentence_level_z / 2:
                    # this is the case that the previous sentence also have low spirit!
                    current_level_z = prev_level_z * 1.1
                elif self.sentence_stats[i].mean_i < self.global_stats.mean_i - 0.5 * self.global_stats.std_i:
                    current_level_z = gen_normal(self.head_sentence_level_z * 0.75, self.head_sentence_level_z * 0.25)
                    if (random.random() < self.left_side_bias):
                        current_level_z *= -1
                else:
                    current_level_z = gen_normal(0, self.head_sentence_level_z * 0.25)

                # add the sharfin point
                sentence_level_head_levels_z.append((current_level_z - sentence_level_head_levels_z[-1]) * 0.8 +
                                                    sentence_level_head_levels_z[-1])
                sentence_level_head_levels_zt.append(sentence_level_transience_head_levels_xt[1] + z_delay)
                # add the point to hold at
                sentence_level_head_levels_z.append(current_level_z)
                sentence_level_head_levels_zt.append(sentence_level_transience_head_levels_xt[2] + z_delay)
                # add the end point
                if self.semantics_script.word_intervals[interval[-1]][1] - 0.4 > sentence_level_head_levels_zt[-1]:
                    sentence_level_head_levels_zt.append(
                        self.semantics_script.word_intervals[interval[-1]][1] - 0.4 + z_delay)
                    sentence_level_head_levels_z.append(current_level_z * 0.8)
                prev_level_z = current_level_z
                # =============== sentence level head tilt when the intensity is lower than average ===============
                # here the level is for nods which is very common
                # add the curve to the list of intervals
                sentence_curve_set.add_to_interval_tier(
                    [sentence_level_transience_head_levels_xt, sentence_level_transience_head_levels_x], 5)
                sentence_curve_set.add_to_interval_tier(
                    [sentence_level_stable_head_levels_xt, sentence_level_stable_head_levels_x], 0)

                sentence_curve_set_z.add_to_interval_tier([sentence_level_head_levels_zt, sentence_level_head_levels_z],
                                                          5)

                # update where the previous big jump in neck amplitude is
                # ====================================== add sentence interval ======================================
            else:
                sentence_level_transience_head_levels_xt.append(self.semantics_script.sentence_intervals[i][0])
                sentence_level_transience_head_levels_xt.append(self.semantics_script.sentence_intervals[i][1])
                sentence_level_transience_head_levels_x.append(prev_level)
                sentence_level_transience_head_levels_x.append(prev_level * 0.8)
                sentence_curve_set.add_to_interval_tier(
                    [sentence_level_transience_head_levels_xt, sentence_level_transience_head_levels_x], 5)
                sentence_level_head_levels_zt.append(self.semantics_script.sentence_intervals[i][0])
                sentence_level_head_levels_zt.append(self.semantics_script.sentence_intervals[i][1])
                sentence_level_head_levels_z.append(prev_level_z)
                sentence_level_head_levels_z.append(prev_level_z * 0.8)
                sentence_curve_set_z.add_to_interval_tier([sentence_level_head_levels_zt, sentence_level_head_levels_z],
                                                          0)
            # ================================= add word level emphasis ============================
            if len(interval) > 2 and False:
                # see if the sentence is negative......
                if len(negative_word_id) == 0:
                    # keep track to see if the emphasis is at the start of the sentence
                    end_of_sentence = False
                    start_of_sentence = False
                    # go through the words and find words that are followed by silences
                    emphasis_tracking = [[], [], [], [], [], []]  # [id of word, silence_before/silence_after/std_intensity/pitch_std/max_intensity/mean_intensity]
                    for j in range(0, len(interval)):
                        distance_from_prev_spike = self.semantics_script.word_intervals[interval[j]][0] - prev_big_spike_time
                        skip = distance_from_prev_spike <= 0.3
                        if self.semantics_script.word_list[interval[j]] != "." and not self.semantics_script.word_list[
                                                                                      interval[j]][
                                                                                  1:] in self.word_not_emphasized:
                            if self.semantics_script.word_list[interval[j] - 1] == "." and j != 1 and not skip:
                                prev_interval = self.semantics_script.word_intervals[interval[j] - 1]
                                # obtain the pause before a word, if there is any
                                length = prev_interval[1] - prev_interval[0]
                                emphasis_tracking[0].append(length)
                            else:
                                emphasis_tracking[0].append(0)
                    for j in range(0, len(interval) - 1):
                        distance_from_prev_spike = self.semantics_script.word_intervals[interval[j]][0] - prev_big_spike_time
                        skip = distance_from_prev_spike <= 0.3
                        if self.semantics_script.word_list[interval[j]] != "." and not self.semantics_script.word_list[
                                                                                      interval[j]][
                                                                                  1:] in self.word_not_emphasized:
                            if self.semantics_script.word_list[interval[j] + 1] == "." and j != len(
                                    interval) - 1 and not skip:
                                next_interval = self.semantics_script.word_intervals[interval[j] + 1]
                                # obtain the pause before a word, if there is any
                                length = next_interval[1] - next_interval[0]
                                emphasis_tracking[1].append(length)
                            else:
                                emphasis_tracking[1].append(0)

                    # go through list of words and find words with large pitch variations/max intensity, and mean intensity
                    per_word_std_pitch = []
                    per_word_max_intensity = []
                    for j in range(0, len(interval)):
                        distance_from_prev_spike = self.semantics_script.word_intervals[interval[j]][0] - prev_big_spike_time
                        skip = distance_from_prev_spike <= 0.5
                        if not skip and self.semantics_script.word_list[interval[j]] != "." and not \
                        self.semantics_script.word_list[interval[j]][1:] in self.word_not_emphasized:
                            emphasis_tracking[2].append(self.word_stats[interval[j]].std_i)
                            emphasis_tracking[3].append(self.word_stats[interval[j]].std_f)
                            emphasis_tracking[4].append(self.word_stats[interval[j]].max_i)
                            emphasis_tracking[5].append(self.word_stats[interval[j]].mean_i)
                        else:
                            emphasis_tracking[2].append(0)
                            emphasis_tracking[3].append(0)
                            emphasis_tracking[4].append(0)
                            emphasis_tracking[5].append(0)
                    # determine the emphasized word based on both pauses and the acoustic properties

                    if np_max_min(emphasis_tracking[0])[0] > 0:

                        post_silence_emphasis_peak = np_argmax(emphasis_tracking[0])
                        prev_silence_emphasis_peak = np_argmax(emphasis_tracking[1])
                        comparison = 0
                        for i in range(3, len(emphasis_tracking)):
                            if emphasis_tracking[post_silence_emphasis_peak, i] > emphasis_tracking[prev_silence_emphasis_peak, i]:
                                comparison += 1
                        if comparison.sum() >= 2:
                            emphasis_peak = post_silence_emphasis_peak + interval[0]
                        else:
                            emphasis_peak = prev_silence_emphasis_peak + interval[0]
                    else:
                        emphasis_ranking_separate = []
                        for iiii in range(3, len(emphasis_tracking)):
                            emphasis_ranking_separate.append(np_softmax(emphasis_tracking[iiii]))
                        emphasis_ranking = []
                        for tttt in range(0, len(emphasis_tracking[3])):
                            val = 0
                            for iiii in range(0, len(emphasis_ranking_separate)):
                                val += emphasis_tracking[iiii, tttt]
                            emphasis_ranking.append(val)
                        emphasis_peak = np_argmax(emphasis_ranking) + interval[0]

                    emphasized_words.append(emphasis_peak)
                    # the next set of key will be the emphasise of the sentence
                    word = self.semantics_script.word_list[emphasis_peak]
                    word_interval = self.semantics_script.word_intervals[emphasis_peak]
                    # determine how much head nod is needed:
                    time_from_prev_big_spike = (word_interval[0] - prev_big_spike_time) / self.time_between_emphasis
                    if emphasis_tracking[:, 0].max() > 0:
                        delta_raise = self.emphasis_max_rotation / 2 + min(
                            (time_from_prev_big_spike) * self.emphasis_max_rotation / 2, 1)
                        delta_drop = self.emphasis_max_rotation / 2 + min(
                            (time_from_prev_big_spike) * self.emphasis_max_rotation * 3 / 2, 1)
                    else:
                        delta_raise = self.emphasis_max_rotation / 2 * 0.6 + min(
                            (time_from_prev_big_spike) * self.emphasis_max_rotation / 2 * 0.6, self.emphasis_max_rotation * 0.3)
                        delta_drop = self.emphasis_max_rotation / 2 + min(
                            (time_from_prev_big_spike) * self.emphasis_max_rotation / 2, self.emphasis_max_rotation / 2)

                    # set up keys
                    emphasis_curve_t = []
                    emphasis_curve_x = []
                    prev_interval = self.semantics_script.word_intervals[emphasis_peak - 1]
                    preparation_key_time = self.semantics_script.word_intervals[emphasis_peak - 1][0] - 0.3
                    preparation_key_val = 0
                    raise_key_time = (prev_interval[0] + prev_interval[1]) / 2
                    raise_key_time = min(raise_key_time, word_interval[0] - 0.25)
                    raise_key_val = preparation_key_val + delta_raise / 2
                    drop_peak_time = word_interval[0]
                    drop_peak_val = raise_key_val - delta_drop
                    drop_hold_time = (word_interval[1] + word_interval[0]) / 2
                    drop_hold_val = drop_peak_val * 0.8
                    drop_end_time = word_interval[1] + 0.2
                    drop_end_val = 0
                    prev_big_spike_time = drop_hold_time

                    # put it into a single interval
                    emphasis_curve_t = [preparation_key_time, raise_key_time, drop_peak_time, drop_hold_time,
                                        drop_end_time]
                    emphasis_curve_x = [preparation_key_val, raise_key_val, drop_peak_val, drop_hold_val, drop_end_val]
                    # add it to the data structure
                    impulse_curve_set.add_to_interval_tier([emphasis_curve_t, emphasis_curve_x], 2)
                else:
                    emphasis_curve_yt = []
                    emphasis_curve_y = []
                    emphasis_tracking = [[], [], [], []]
                    for word_id in range(len(negative_word_id)):
                        emphasis_tracking[0].append(self.word_stats[negative_word_id[word_id]].std_i)
                        emphasis_tracking[1].append(self.word_stats[negative_word_id[word_id]].std_f)
                        emphasis_tracking[2].append(self.word_stats[negative_word_id[word_id]].max_i)
                        emphasis_tracking[3].append(self.word_stats[negative_word_id[word_id]].mean_i)
                    # compute softmax on each feature and sum them up
                    emphasis_ranking_separate = []
                    for iiii in range(3, len(emphasis_tracking)):
                        emphasis_ranking_separate.append(np_softmax(emphasis_tracking[iiii]))
                    emphasis_ranking = []
                    for tttt in range(0, len(emphasis_tracking[3])):
                        val = 0
                        for iiii in range(0, len(emphasis_ranking_separate)):
                            val += emphasis_tracking[iiii, tttt]
                        emphasis_ranking.append(val)
                    emphasis_peak = negative_word_id[np_argmax(emphasis_ranking)] + interval[0]
                    word_interval = self.semantics_script.word_intervals[emphasis_peak]
                    number_of_shake = self.base_number_of_shakes + random.randint(0, int(self.base_number_of_shakes / 2)) - math.ceil(self.base_number_of_shakes / 4)

                    emphasis_curve_yt.append(word_interval[0] - 0.4)
                    emphasis_curve_y.append(0)
                    starting_sign = 1
                    if (random.random() < self.left_side_bias):
                        starting_sign = -1
                    for i in range(0, number_of_shake):
                        delta_shake = self.emphasis_max_rotation / 4 + gen_normal(0, self.emphasis_max_rotation)
                        emphasis_curve_yt.append(emphasis_curve_yt[-1] + 0.2)
                        emphasis_curve_y.append(
                            delta_shake * starting_sign * (((number_of_shake - i) / number_of_shake) ** 3))
                        starting_sign *= -1
                    emphasis_curve_yt.append(emphasis_curve_yt[-1] + 0.1)
                    emphasis_curve_y.append(0)
                    impulse_curve_set_y.add_to_interval_tier([emphasis_curve_yt, emphasis_curve_y], 2)
        # ================================= add word level emphasis ============================
        # add in the jitter tier motion
        # ================================= add jitter tier ambience ============================
        prev_motion = 0
        for i in range(0, len(self.semantics_script.word_to_phone)):
            if self.semantics_script.word_list[i] == ".":
                continue
            if negative_sentences[self.semantics_script.word_to_sentence[i]]:
                continue
            interval = self.semantics_script.word_to_phone[i]
            emphasis_curve_t = []
            emphasis_curve_x = []
            # avoid adding jitter to a word in the beginning of a sentence
            current_sentence_id = self.semantics_script.word_to_sentence[i]
            beginning_of_sentence = (self.semantics_script.sentence_to_word[current_sentence_id][0] == i)
            end_of_sentence = self.semantics_script.sentence_to_word[current_sentence_id][-1] == i
            emphasized_already = (i in emphasized_words)
            if beginning_of_sentence or emphasized_already or end_of_sentence:
                continue
            for j in range(0, len(interval)):
                if self.semantics_script.phone_list[interval[j]][-1] == "1":
                    from_prev = self.semantics_script.phone_intervals[interval[j]][0] - prev_motion
                    phone_interval = self.semantics_script.phone_intervals[interval[j]]
                    delta_raise = self.max_ambient_movement
                    # check to see if the interval intersect with a key sentence interval
                    if from_prev >= self.threshold_from_previous_jitter:
                        test_with_interval = SplineInterval(-1, phone_interval[0] - 0.35, phone_interval[1] + 0.2, [-1],
                                                            0)
                        not_on_transition = True
                        for priority_interval in sentence_curve_set.impulse_intervals:
                            if priority_interval.priority == 5 and priority_interval.intersect(test_with_interval):
                                not_on_transition = False
                                break
                        for priority_interval in impulse_curve_set.impulse_intervals:
                            if priority_interval.intersect(test_with_interval):
                                not_on_transition = False
                                break
                        if not_on_transition:
                            preparation_key_time = phone_interval[0] - 0.35
                            prev_interval = self.semantics_script.phone_intervals[max(interval[j] - 1, 0)]
                            preparation_key_val = 0
                            raise_key_time = (prev_interval[0] + prev_interval[1]) / 2
                            raise_key_time = min(raise_key_time, phone_interval[0] - 0.20)
                            raise_key_val = preparation_key_val + delta_raise / 2
                            drop_peak_time = phone_interval[0]
                            drop_peak_val = raise_key_val - delta_raise
                            drop_hold_time = (phone_interval[1] + phone_interval[0]) / 2
                            drop_hold_val = drop_peak_val * 0.8
                            drop_end_time = phone_interval[1] + 0.2
                            drop_end_val = 0
                            prev_motion = drop_hold_time
                            emphasis_curve_t = [preparation_key_time, raise_key_time, drop_peak_time, drop_end_time]
                            emphasis_curve_x = [preparation_key_val, raise_key_val, drop_peak_val, drop_end_val]
                            # print(emphasis_curve_t, emphasis_curve_x)
                            jitter_curve_set.add_to_interval_tier([emphasis_curve_t, emphasis_curve_x], 1)
                            break
        # ================================= add jitter tier ambience ============================
        # output to maya
        sentence_curve_set.fill_holes()

        jitter_xt, jitter_x = jitter_curve_set.recompute_all_points()
        impulse_xt, impulse_x = impulse_curve_set.recompute_all_points()
        sentence_level_xt, sentence_level_x = sentence_curve_set.recompute_all_points()
        out_xt, out_x = sentence_curve_set.sum_with_other_tiers([impulse_curve_set, jitter_curve_set])
        out_x = sparse_key_smoothing(out_xt, out_x, smoothing_win_size=3)
        out_zt, out_z = sentence_curve_set_z.recompute_all_points(mean=True)
        out_z = sparse_key_smoothing(out_zt, out_z, smoothing_win_size=3)
        out_yt, out_y = impulse_curve_set_y.recompute_all_points(mean=True)
        output = [out_xt, out_x, out_yt, out_y, out_zt, out_z]
        outputAsyncToFile(self.out_path, output)
        return output


if __name__ == "__main__":
    k = NeckCurve("C:/ProgramData/Jali/resources/input/Merchant_Intro/merchant1.wav")
    out = k.compute_curve()
    print(out)
    A[2]


    file_path = "C:/Users/evan1/Documents/neckMovement/data/neck_rotation_values/"
    # file_path = "F:/MASC/JALI_neck/data/neck_rotation_values/"
    file_names = ["sarah.mp4", "news_anchor_1.mov", "Merchant_Intro.mp3", "Freewill_Tags.mp3", "Uther.mp3",
                  "Arthus.mp3", "Skyler.mp3", "not_ur_fault.mp3"]
    file_name = file_names[2]
    NeckCurve("C:/ProgramData/Jali/resources/input/Merchant_Intro/merchant1.wav")
