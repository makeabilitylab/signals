import os
import numpy as np 
from gesturerec.data import SensorData

def get_gesture_stream_with_str(map_gesture_streams, s):
    '''
    Gets the gesture set containing the str s 
    '''
    for base_path, gesture_stream in map_gesture_streams.items():
        if s in base_path:
            return gesture_stream
    return None

class GestureStream:
    '''
    The primary data structure to analyze the full datastream (offline)
    '''
    FULL_DATASTREAM_FILENAME = 'fulldatastream.csv'
    
    def __init__(self, path_to_gesture_log):
        self.path = path_to_gesture_log
        path_to_full_sensor_stream_file = os.path.join(path_to_gesture_log, GestureStream.FULL_DATASTREAM_FILENAME)
        self.filename_with_path = path_to_full_sensor_stream_file
        self.name = self.__get_base_path() # do not change the name, it's used as an dict key
    
    def load(self):
        self.sensor_stream = self.__parse_full_sensor_stream(self.filename_with_path) # SensorData object
    
    def __parse_full_sensor_stream(self, path_to_file):
        
        # fix and verify full sensor stream file
        self.__verify_and_fix_full_sensor_stream(path_to_file)
        
        # parsing 
        parsed_log_data = np.genfromtxt(path_to_file, dtype='str', delimiter=',', 
                                        encoding=None, skip_header=0, 
                                        unpack=True, invalid_raise = False)

        full_sensor_stream = SensorData("Accelerometer", *parsed_log_data)
        return full_sensor_stream
        
    def get_index_closest_to_timestamp(self, timestamp):
        '''
        Returns the closest sensor stream row index closest to timestamp
        '''
        # from https://stackoverflow.com/a/26026189
        idx = np.searchsorted(self.sensor_stream.time, timestamp, side="left")
        return idx
    
    def __verify_and_fix_full_sensor_stream(self, path_to_file):
        '''
        Sometimes the fulldatastream.csv has some small errors in it. This function looks for 
        those errors, ignores those rows, and saves a new 'clean' file without errors
        '''
        import csv
        
        TIME_COL_IDX = 0
        SENSOR_TIME_COL_IDX = 1
        X_IDX = 2
        Y_IDX = 3
        Z_IDX = 4
        MAX_COLUMNS = 5
        
        row_idx = 1
        list_rows = []
        problem_cnt = 0
        
        # The following loop fixes problem lines and removes them
        # For example, in Jesse's log, he has a line (line 4214) that has a *huge* anamalous timestamp:
        # 1571573159855494	8290	337	342	413
        with open(path_to_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            try:
                for row in csv_reader:
                    is_good_row = True
                    
                    if len(row) > MAX_COLUMNS:
                        print("WARNING: Row {} has more than {} columns. The full row:".format(row_idx, MAX_COLUMNS))
                        print(', '.join(row))
                        is_good_row = False
                        problem_cnt += 1
                    
                    col_idx = 0
                    for col in row:
                        stripped_col = col.strip()
                        if not stripped_col: # check for empty str https://stackoverflow.com/a/9573283
                            print("WARNING: Row {} Col {} is empty. The full row:".format(row_idx, col_idx))
                            print(', '.join(row))
                            is_good_row = False
                            problem_cnt += 1
                        col_idx += 1
                            
                    
                    raw_time = int(row[TIME_COL_IDX])
                    sensor_time = int(row[SENSOR_TIME_COL_IDX])
                    x = int(row[X_IDX])
                    y = int(row[Y_IDX])
                    z = int(row[Z_IDX])
                    
                    if is_good_row:
                        list_rows.append(row)
                    
                    row_idx += 1
                
            except Exception as e:
                print("Row {} Error: {}".format(row_idx, e))
                problem_cnt += 1
        
        if problem_cnt > 0:
            import ntpath, os, time
            
            print("File '{}' contained {} lines and {} problems".format(path_to_file, row_idx - 1, problem_cnt))
            path = os.path.dirname(os.path.abspath(path_to_file))
            cur_filename = ntpath.basename(path_to_file)
            cur_filename_without_ext = os.path.splitext(cur_filename)[0]
            new_filename = '{}_old_{}.csv'.format(cur_filename_without_ext, int(round(time.time() * 1000)))
            new_filename_with_path = os.path.join(path, new_filename)
            os.rename(path_to_file, new_filename_with_path)
            print("Renamed problem file to '{}'".format(new_filename_with_path))
            
            with open(path_to_file, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerows(list_rows)
            
            print("Wrote {} 'cleaned' rows to '{}'".format(len(list_rows), path_to_file))
        else:
            print("Successfully checked all {} rows in '{}'. No problems found".format(row_idx-1, path_to_file))
    
    def __get_base_path(self):
        return os.path.basename(os.path.normpath(self.path))
    
    def __str__(self):
         return "'{}' : {} rows {:0.1f} secs {:.2f} Hz".format(self.filename_with_path, len(self.sensor_stream.time),
                                                      self.sensor_stream.length_in_secs,
                                                      self.sensor_stream.sampling_rate)
                                                       
class Event:
    '''
    A segmented event in the sensor stream. Similar to a Trial object, contains a
    SensorData object that holds all of the sensor data for this Event
    '''
    def __init__(self, event_idx, sensor_stream, stream_start_idx, stream_end_idx):
        self.event_idx = event_idx
        self.start_idx = stream_start_idx
        self.end_idx = stream_end_idx
        self.start_timestamp = sensor_stream.time[stream_start_idx]
        self.end_timestamp = sensor_stream.time[stream_end_idx]
        self.length_ms = self.end_timestamp - self.start_timestamp
        self.associated_ground_truth_trial = None
           
        # (self, sensor_type, time, sensor_time, x, y, z):
        t = sensor_stream.time[stream_start_idx:stream_end_idx]
        sensor_t = sensor_stream.sensor_time[stream_start_idx:stream_end_idx]
        x = sensor_stream.x[stream_start_idx:stream_end_idx]
        y = sensor_stream.y[stream_start_idx:stream_end_idx]
        z = sensor_stream.z[stream_start_idx:stream_end_idx]
        self.accel = SensorData(sensor_stream.sensor_type, t, sensor_t, x, y, z)
    
    def get_ground_truth_gesture_name(self):
        '''If a ground truth trial has been associated with this Event, return its gesture name'''
        if self.associated_ground_truth_trial == None:
            return 'Null'
        else:
            return self.associated_ground_truth_trial.gesture_name
    
    def length(self):
        '''Gets the length in samples'''
        return len(self.accel.x)
    
    def get_start_time(self):
        '''Gets the start time'''
        return self.accel.time[0]
    
    def get_end_time(self):
        '''Gets the end time'''
        return self.accel.time[-1]
        
    def __str__(self):
        return 'Event #{}: start_idx={} end_idx={} {} ms'.format(self.event_idx,
                                                                         self.start_idx, self.end_idx,
                                                                         self.end_timestamp - self.start_timestamp)