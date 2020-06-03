# This cell includes the major classes used in our classification analyses
# import matplotlib.pyplot as plt # needed for plotting
import numpy as np # numpy is primary library for numeric array (and matrix) handling
# import scipy as sp
# from scipy import signal
import random
import os
import gesturerec.utility

class SensorData:
    '''
    Contains the gyroscope, accelerometer, or other sensor data as numpy arrays
    '''
     
    def __init__(self, sensor_type, time, sensor_time, x, y, z):
        '''
        All arguments are numpy arrays except sensor_type, which is a str
        '''
        self.sensor_type = sensor_type
        
        # On my mac, I could cast as straight-up int but on Windows, this failed
        # This is because on Windows, a long is 32 bit but on Unix, a long is 64bit
        # So, forcing to int64 to be safe. See: https://stackoverflow.com/q/38314118
        self.time = time.astype(np.int64) # timestamps are in milliseconds
        
        # sensor_time comes from the Arduino function. it's in milliseconds 
        # https://www.arduino.cc/reference/en/language/functions/time/millis/
        # which returns the number of milliseconds passed since the Arduino board began running the current program.
        self.sensor_time = sensor_time.astype(np.int64) # timestamps are in milliseconds
        
        self.x = x.astype(float)
        self.y = y.astype(float)
        self.z = z.astype(float)
   
        # Calculate the magnitude of the signal
        self.mag = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        
        # Create placeholders for processed data
        self.x_p = None
        self.y_p = None
        self.z_p = None
        self.mag_p = None

        self.length_in_secs = (self.time[-1] - self.time[0]) / 1000.0
        self.sampling_rate = len(self.time) / self.length_in_secs 
        
    def length(self):
        '''
        Returns length (in rows). Note that all primary data structures: time, x, y, z, and mag
        are the same length. So just returns len(self.x). Depending on the preprocessing alg,
        the processed data may be a different length than unprocessed
        '''
        return len(self.x)
        
    def get_data(self):
        '''
        Returns a dict of numpy arrays for each axis of the accel + magnitude
        '''
        return {"x":self.x, "y":self.y, "z":self.z, "mag":self.mag}
    
    def get_processed_data(self):
        '''
        Returns a dict of numpy arrays for each axis of the accel + magnitude
        '''
        return {"x_p":self.x_p, "y_p":self.y_p, "z_p":self.z_p, "mag_p":self.mag_p}

    def __str__(self):
        return "{}: {} samples {:.2f} secs {:.2f} Hz".format(self.sensor_type, self.length(),
                                                    self.length_in_secs, self.sampling_rate)


class Trial:
    '''
    A trial is one gesture recording and includes an accel SensorData object
    In the future, this could be expanded to include other recorded sensors (e.g., a gyro)
    that may be recorded simultaneously
    '''
    
    def __init__(self, gesture_name, trial_num, log_filename_with_path):
        '''
        We actually parse the sensor log files in the constructor--this is probably bad practice
        But offers a relatively clean solution
        
        gesture_name : the gesture name (as a str)
        trial_num : the trial number (we asked you to collect 5 or maybe 10 trials per gesture)
        log_filename_with_path : the full path to the filename (as a str)
        '''
        self.gesture_name = gesture_name
        self.trial_num = trial_num
        self.log_filename_with_path = log_filename_with_path
        self.log_filename = os.path.basename(log_filename_with_path)
        
        # unpack=True puts each column in its own array, see https://stackoverflow.com/a/20245874
        # I had to force all types to strings because auto-type inferencing failed
        parsed_accel_log_data = np.genfromtxt(log_filename_with_path, delimiter=',', 
                              dtype=str, encoding=None, skip_header=1, unpack=True)
        
        # The asterisk is really cool in Python. It allows us to "unpack" this variable
        # into arguments needed for the SensorData constructor. Google for "tuple unpacking"
        self.accel = SensorData("Accelerometer", *parsed_accel_log_data)
    
    def get_ground_truth_gesture_name(self):
        '''Returns self.gesture_name'''
        return self.gesture_name
        
    def length(self):
        '''Gets the length of the trial in samples'''
        return len(self.accel.x)
    
    def get_start_time(self):
        '''Gets the start timestamp'''
        return self.accel.time[0]
    
    def get_end_time(self):
        '''Gets the end timestamp'''
        return self.accel.time[-1]
    
    def get_end_time_as_string(self):
        '''Utility function that returns the end time as a nice string'''
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.get_end_time() / 1000))
    
    def __str__(self):
         return "'{}' : Trial {} from {}".format(self.gesture_name, self.trial_num, self.log_filename)
        
class GestureSet:
    '''
    Container for a single set of gestures and trials
    '''

    DEFAULT_GESTURE_NAMES = set(['At Rest', 'Backhand Tennis', 'Baseball Throw', 
                          'Custom', 'Forehand Tennis', "Midair 'S'", 
                          "Midair Clockwise 'O'", "Midair Counter-clockwise 'O'", 
                          "Midair Zorro 'Z'", 'Shake', 'Underhand Bowling'])
    
    GESTURE_NAMES_WITHOUT_CUSTOM = None
    
    def __init__(self, gesture_log_path):
        '''
        After calling the constructor, you must call *load* and then *preprocess*
        
        Parameters:
        gesture_log_path: path to the gesture log dir
        '''
        self.path = gesture_log_path
        self.name = self.get_base_path() # do not change the name, it's used as an dict key

        self.GESTURE_NAMES_WITHOUT_CUSTOM = set(self.DEFAULT_GESTURE_NAMES)
        self.GESTURE_NAMES_WITHOUT_CUSTOM.remove('Custom')
        
    def load(self):
        '''Loads the gesture trials.'''
        
        # Our primary object tha maps a gesture name to a list of Trial objects
        self.map_gestures_to_trials = self.__parse_gesture_trials(self.path)   
    
    def __parse_gesture_trials(self, path_to_dir):
        '''
        Parses and creates Trial objects for all csv files in the given dir. 
        It's not necessary that you understand this code
        
        Parameters:
        path_to_dir: the path to the gesture logs
        
        Returns:
        dict: a dict() mapping (str: gesture_name) to (list: Trial objects)
        '''
        csv_filenames = gesturerec.utility.find_csv_filenames(path_to_dir)

        print("Found {} csv files in {}".format(len(csv_filenames), path_to_dir))

        map_gesture_name_to_trial_list = dict()
        map_gesture_name_to_map_endtime_to_map_sensor_to_file = dict() # use this to correctly order trials
        for csvFilename in csv_filenames:

            # parse filename into meaningful parts
            # print(csvFilename)
            filename_no_ext = os.path.splitext(csvFilename)[0];

            filename_parts = filename_no_ext.split("_")
            gesture_name = None
            time_ms = None
            num_rows = None
            sensor_name = "Accelerometer" # currently only one sensor but could expand to more

            # Added this conditional because Windows machines created differently formatted
            # filenames from Macs. Windows machines automatically replaced the character "'"
            # with "_", which affects filenames like "Midair Zorro 'Z'_1556730840228_206.csv"
            # which come out like "Midair Zorro _Z__1557937136974_211.csv" instead
            if '__' in filename_no_ext:
                filename_parts1 = filename_no_ext.split("__")
                gesture_name = filename_parts1[0]
                gesture_name = gesture_name.replace('_',"'")
                gesture_name += "'"

                filename_parts2 = filename_parts1[1].split("_")
                time_ms = filename_parts2[0]
                num_rows = filename_parts2[1]
            else:
                filename_parts = filename_no_ext.split("_")
                gesture_name = filename_parts[0]
                time_ms = filename_parts[1]
                num_rows = int(filename_parts[2])

            # print("gesture_name={} time_ms={} num_rows={}".format(gesture_name, time_ms, num_rows))

            if gesture_name not in map_gesture_name_to_map_endtime_to_map_sensor_to_file:
                map_gesture_name_to_map_endtime_to_map_sensor_to_file[gesture_name] = dict()

            if time_ms not in map_gesture_name_to_map_endtime_to_map_sensor_to_file[gesture_name]:
                map_gesture_name_to_map_endtime_to_map_sensor_to_file[gesture_name][time_ms] = dict()

            map_gesture_name_to_map_endtime_to_map_sensor_to_file[gesture_name][time_ms][sensor_name] = csvFilename
            # print (map_gesture_name_to_map_endtime_to_map_sensor_to_file)

        print("Found {} gestures".format(len(map_gesture_name_to_map_endtime_to_map_sensor_to_file)))

        # track the longest array
        max_array_length = -1
        trial_with_most_sensor_events = None

        # Now we need to loop through the data and sort each gesture set by timems values 
        # (so that we have trial 1, 2, 3, etc. in order)
        for gesture_name, map_endtime_to_map_sensor_to_file in map_gesture_name_to_map_endtime_to_map_sensor_to_file.items():
            gesture_trial_num = 0
            map_gesture_name_to_trial_list[gesture_name] = list()
            for end_time_ms in sorted(map_endtime_to_map_sensor_to_file.keys()):
                map_sensor_to_file = map_endtime_to_map_sensor_to_file[end_time_ms]

                log_filename_with_path = os.path.join(path_to_dir, map_sensor_to_file["Accelerometer"])
                gesture_trial = Trial(gesture_name, gesture_trial_num, log_filename_with_path)
                map_gesture_name_to_trial_list[gesture_name].append(gesture_trial)

                if max_array_length < len(gesture_trial.accel.x):
                    max_array_length = len(gesture_trial.accel.x)
                    trial_with_most_sensor_events = gesture_trial

                gesture_trial_num = gesture_trial_num + 1

            print("Found {} trials for '{}'".format(len(map_gesture_name_to_trial_list[gesture_name]), gesture_name))

        # Print out some basic information about our logs
        print("Max trial length across all gesture is '{}' Trial {} with {} sensor events.".
              format(trial_with_most_sensor_events.gesture_name, trial_with_most_sensor_events.trial_num, max_array_length))
        list_samples_per_second = list()
        list_total_sample_time = list()
        for gesture_name, trial_list in map_gesture_name_to_trial_list.items():
            for trial in trial_list: 
                list_samples_per_second.append(trial.accel.sampling_rate)
                list_total_sample_time.append(trial.accel.length_in_secs)

        print("Avg samples/sec across {} sensor files: {:0.1f}".format(len(list_samples_per_second), 
                                                                       sum(list_samples_per_second)/len(list_samples_per_second)))
        print("Avg sample length across {} sensor files: {:0.1f}s".format(len(list_total_sample_time), 
                                                                          sum(list_total_sample_time)/len(list_total_sample_time)))
        print()
        return map_gesture_name_to_trial_list
    
    def get_trials(self, gesture_name):
        '''Returns a list of trials for this gesture name sorted chronologically'''
        return self.map_gestures_to_trials[gesture_name]
    
    def get_all_trials(self):
        '''Gets all trials sorted chronologically'''
        trials = list()
        for gesture_name, trial_list in self.map_gestures_to_trials.items():
            trials += trial_list
            
        trials.sort(key=lambda x: x.get_start_time())
        return trials
    
    def get_all_trials_except(self, trial):
        '''Gets all the trials except the given trial'''
        trials = self.get_all_trials()
        trials.remove(trial)
        return trials     
    
    def get_trials_that_overlap(self, start_timestamp, end_timestamp):
        '''Returns the trials that overlap the start and end timestamps (inclusive)'''
        matched_trials = list()
        trials = self.get_all_trials()
        for trial in trials:
            if trial.get_end_time() >= start_timestamp and trial.get_start_time() <= end_timestamp:
                matched_trials.append(trial)
            elif trial.get_start_time() > end_timestamp:
                break # trials are ordered, no need to continue through list
        return matched_trials
    
    def get_longest_trial(self):
        '''Returns the longest trial (based on num rows recorded)'''
        longest_trial_length = -1
        longest_trial = None
        for gesture_name, trial_list in self.map_gestures_to_trials.items():
            for trial in trial_list:
                if longest_trial_length < len(trial.accel.x):
                    longest_trial_length = len(trial.accel.x)
                    longest_trial = trial
        return longest_trial
    
    def get_base_path(self):
        '''Returns the base path of self.path'''
        return os.path.basename(os.path.normpath(self.path))
    
    def get_num_gestures(self):
        '''Returns the number of gestures'''
        return len(self.map_gestures_to_trials)
    
    def get_trials_for_gesture(self, gesture_name):
        '''Returns trials for the given gesture name'''
        return self.map_gestures_to_trials[gesture_name]
        
    def get_min_num_of_trials(self):
        '''
        Returns the minimum number of trials across all gestures (just in case we accidentally recorded a 
        different number. We should have the same number of trials across all gestures)
        '''
        min_num_trials = -1 
        for gesture_name, trials in self.map_gestures_to_trials.items():
            if min_num_trials == -1 or min_num_trials > len(trials):
                min_num_trials = len(trials)
        return min_num_trials

    def get_total_num_of_trials(self):
        '''Returns the total number of trials'''
        numTrials = 0 
        for gesture_name, trialSet in self.map_gestures_to_trials.items():
            numTrials = numTrials + len(trialSet)
        return numTrials
    
    def get_random_gesture_name(self):
        '''Returns a random gesture name from within this GestureSet'''
        gesture_names = list(self.map_gestures_to_trials.keys())
        rand_gesture_name = gesture_names[random.randint(0, len(gesture_names) - 1)]
        return rand_gesture_name
    
    def get_random_trial_for_gesture(self, gesture_name):
        '''Returns a random trial for the given gesture name'''
        trials_for_gesture = self.map_gestures_to_trials[gesture_name]
        return trials_for_gesture[random.randint(0, len(trials_for_gesture) - 1)]
    
    def get_random_trial(self):
        '''Returns a random trial'''
        rand_gesture_name = self.get_random_gesture_name()
        print("rand_gesture_name", rand_gesture_name)
        trials_for_gesture = self.map_gestures_to_trials[rand_gesture_name]
        return trials_for_gesture[random.randint(0, len(trials_for_gesture) - 1)]
    
    def get_gesture_names_sorted(self):
        '''Returns a sorted list of gesture names'''
        return sorted(self.map_gestures_to_trials.keys())

    def get_gesture_names_filtered(self, filter_names):
        '''Returns the gesture names except for those in the filter_names list'''
        filter_names = set(filter_names)
        gesture_names = list()
        for gesture_name in self.map_gestures_to_trials.keys():
            if gesture_name not in filter_names:
                gesture_names.append(gesture_names)
        
        return sorted(gesture_names)
    
    def __str__(self):
         return "'{}' : {} gestures and {} total trials".format(self.path, self.get_num_gestures(), 
                                                                self.get_total_num_of_trials())

# Gesture set utility functions

# Gets a random gesture set
def get_random_gesture_set(map_gesture_sets):
    '''
    Returns a random gesture set
    '''
    gesture_set_names = list(map_gesture_sets.keys())
    rand_gesture_set_name = gesture_set_names[random.randint(0, len(gesture_set_names) - 1)]
    return map_gesture_sets[rand_gesture_set_name]

def get_gesture_set(map_gesture_sets, key):
    '''
    Gets the gesture set for the given key
    '''
    return map_gesture_sets[key]

def get_gesture_set_with_str(map_gesture_sets, s):
    '''
    Gets the gesture set containing the str s 
    '''
    for gesture_set_name, gesture_set in map_gesture_sets.items():
        if s in gesture_set_name:
            return gesture_set
    
    print(f"We could not find '{s}' in map_gesture_sets")

    return None

def get_gesture_sets_with_str(map_gesture_sets, s):
    '''
    Gets all gesture sets with s in the name
    
    s: can be a string or a collection of strings
    '''
    gesture_sets = []
    for base_path, gesture_set in map_gesture_sets.items():
        if isinstance(s, str):
            if s in base_path:
                gesture_sets.append(gesture_set)
        else:
            for i_str in s:
                if i_str in base_path:
                    gesture_sets.append(gesture_set)

    if len(gesture_sets) <= 0:
        print(f"We found no gesture sets with the string '{s}'")
    
    return gesture_sets

def get_random_gesture_set(map_gesture_sets):
    '''
    Returns a random gesture set
    '''
    import random
    keys = list(map_gesture_sets.keys())
    rand_key = random.choice(keys)
    rand_gesture_set = map_gesture_sets[rand_key]
    return rand_gesture_set

def get_gesture_set_names_sorted(map_gesture_sets):
    '''
    Returns a list of gesture set names sorted by name
    '''
    return sorted(list(map_gesture_sets.keys()))

def get_all_gesture_sets(map_gesture_sets):
    '''
    Gets all of the gesture sets
    '''
    return map_gesture_sets.values()

def get_all_gesture_sets_except(map_gesture_sets, filter):
    '''
    Gets all of the gesture sets except filter. Filter can be a string
    or a list of strings
    '''
    if isinstance(filter, str):
        filter = [filter]
    
    gesture_sets = []
    for gesture_set_name, gesture_set in map_gesture_sets.items():
        if filter.count(gesture_set_name) <= 0:
            gesture_sets.append(gesture_set)

    return gesture_sets