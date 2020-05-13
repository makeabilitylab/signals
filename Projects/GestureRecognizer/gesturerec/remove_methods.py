def preprocess(self, max_array_length = -1):
        '''
        Preprocesses all of the signals (x, y, z, and mag) and stores them in
        self.x_p, self.y_p, self.z_p, and self.mag_p
        '''
        self.x_p = self.__preprocess(self.x, max_array_length)
        self.y_p = self.__preprocess(self.y, max_array_length)
        self.z_p = self.__preprocess(self.z, max_array_length)
        self.mag_p = self.__preprocess(self.mag, max_array_length)
    
        # JonTODO delete this for posting assignment
        # Need to work on this more
        # length_before_trim = len(self.mag_p)
        # self.__trim()
        #print("Trimmed processed signal from {} samples to {}".format(length_before_trim, len(self.mag_p)))
        
    def __trim(self):
        window_start_idx = 0
        last_window_segment = None
        find_start_window_size = 20
        find_end_window_size = 100
        min_max_threshold = 30
        
        window_size = find_start_window_size
        window_step = 10
        new_start_idx = None
        new_end_idx = None
        while window_start_idx + window_size < len(self.mag_p):
            window_segment_mag_p = self.mag_p[window_start_idx : window_start_idx + window_size]
            min_max_diff = abs(np.max(window_segment_mag_p) - np.min(window_segment_mag_p))
            
            if min_max_diff > min_max_threshold:
                if new_start_idx is None:
                    new_start_idx = window_start_idx # start location of event
                    window_size = find_end_window_size
            else:
               new_end_idx = window_start_idx 
                
            window_start_idx += window_step
        
        self.x_p = self.x_p[new_start_idx:new_end_idx]
        self.y_p = self.y_p[new_start_idx:new_end_idx]
        self.z_p = self.z_p[new_start_idx:new_end_idx]
        self.mag_p = self.mag_p[new_start_idx:new_end_idx]
        # JonTODO end delete
    
    def __preprocess(self, raw_signal, max_array_length = -1):
        '''
        Private function to preprocess all of the data
        '''
        # CSE599TODO: You'll want to loop through the sensor signals and preprocess them
        # Some things to explore: padding signal to equalize length between trials, smoothing, detrending, and scaling
        # Another thing you could explore is to trim the trials so they are as tight as possible
        # around the 'event' itself
  
        # For example, this code smooths the signal using a median filter 
        # https://en.wikipedia.org/wiki/Median_filter#Worked_1D_example
        # Note: a mean filter would work better :)
        
        # Uncomment the following for the assignment
        # med_filter_window_size = 9
        # processed_signal = signal.medfilt(raw_signal, med_filter_window_size)
        # 
        # return processed_signal
        
        
        # *****JonTODO: Delete the following when posting assignment
        # Detrend the signal
        processed_signal = sp.signal.detrend(raw_signal) 
        
        # Smooth the signal with a sliding window average filter
        mean_filter_window_size = 10
        processed_signal = np.convolve(processed_signal, np.ones((mean_filter_window_size,))/mean_filter_window_size, 
                                       mode='valid')
        
        # Pad the signal so that they are all the same length
        if max_array_length != -1:
            array_length_diff = max_array_length - len(raw_signal)
            
            # np.pad allows us to pad either the left side, right side, or both sides of an array
            # in this case, we are padding only the right side. 
            # See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
            processed_signal = np.pad(processed_signal, (0, array_length_diff), 'mean') 
            
        return processed_signal
        # *****end delete