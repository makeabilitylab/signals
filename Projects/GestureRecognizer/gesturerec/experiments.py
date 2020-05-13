# Classification experiment bookkeeping classes
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
import random
import time
import numpy as np

class Experiments:
    '''
    Convenience class to store multiple ClassificationResults
    '''
    def __init__(self):
        self.experiments = [] # list of ClassificationResults objects
    
    def add(self, classification_results):
        '''Adds a ClassificationResult object'''
        self.experiments.append(classification_results)
    
    def get_top_performing_experiment(self):
        '''Returns the top performing experiment'''
        return self.get_experiments_sorted_by_accuracy()[-1]
    
    def get_experiments_sorted_by_accuracy(self, reverse=False):
        '''Returns a list of experiments sorted by accuracy'''
        return sorted(self.experiments, reverse=reverse, key=lambda x: x.get_accuracy())

    def get_experiments_sorted_by_computation_time(self, reverse=False):
        '''Returns a list of experiments sorted by computation time'''
        return sorted(self.experiments, reverse=reverse, key=lambda x: x.total_time)

    def get_experiments_sorted_by_avg_time_per_comparison(self, reverse=False):
        '''Returns a list of experiments sorted by avg time per comparison'''
        return sorted(self.experiments, reverse=reverse, key=lambda x: x.get_avg_time_per_comparison())
    
    def get_experiment_titles(self):
        '''Returns a list of experiment titles (sorted by accuracy)'''
        experiment_names = [experiment.experiment_title for experiment in self.get_experiments_sorted_by_accuracy()]
        return experiment_names
    
    def get_experiment_accuracies(self):
        '''Returns a list of experiment accuracies (sorted by accuracy)'''
        accuracies = [experiment.get_accuracy() for experiment in self.get_experiments_sorted_by_accuracy()]
        return accuracies

    def get_avg_accuracy_with_std(self):
        '''Returns a tuple of (average accuracy, standard deviation)'''
        accuracies = np.array(self.get_experiment_accuracies())
        return (np.mean(accuracies), np.std(accuracies))
    
    def print_results(self):
        '''Prints all results sorted by accuracy'''
        for prediction_result in self.get_experiments_sorted_by_accuracy():
            print(prediction_result.get_title())


class ClassificationResults:
    '''
    Stores results for a classification experiment.
    This is the primary object returned from the function run_matching_algorithm
    '''
    
    def __init__(self, matching_alg_name, map_gesture_name_to_list_results, **kwargs):
        '''
        
        Parameters:
        matching_alg_name: the name of the matching alg used for result
        map_gesture_name_to_list_results: a map of gesture name to list of TrialClassificationResults
        '''
        self.matching_alg_name = matching_alg_name
        self.map_gesture_name_to_list_results = map_gesture_name_to_list_results
    
        self.total_time = 0 # in seconds
        self.total_num_comparisons = 0
        self.total_correct = 0
        self.title = "No title yet"
        self.kwargs = kwargs # optional args
        
        for gesture_name, list_results in map_gesture_name_to_list_results.items():
            self.total_num_comparisons += len(list_results)
            for result in list_results: 
                self.total_time += result.elapsed_time
                if result.is_correct:
                    self.total_correct += 1
    
    def get_avg_time_per_comparison(self):
        '''Returns the average time per comparison'''
        return self.total_time / self.total_num_comparisons
    
    def get_gesture_names(self):
        '''Returns a sorted list of gesture names'''
        return sorted(self.map_gesture_name_to_list_results.keys())
    
    def get_accuracy(self):
        '''Returns the accuracy (which is number correct over number total comparisons)'''
        return self.total_correct / self.total_num_comparisons
   
    def get_num_correct_for_gesture(self, gesture_name):
        '''Returns the number correct for this gesture'''
        list_results_for_gesture = self.map_gesture_name_to_list_results[gesture_name]
        correctness_cnt = 0
        for result in list_results_for_gesture:
            if result.is_correct:
                correctness_cnt += 1
                
        return correctness_cnt
    
    def get_title(self):
        '''Returns the title of this instance'''
        return "{}: {}/{} ({:0.2f}%)"\
              .format(self.title, self.total_correct, self.total_num_comparisons, self.get_accuracy() * 100)

    def get_correct_match_scores_for_gesture(self, gesture_name):
        '''Returns a list of scores for the correct matches for this gesture'''
        results_for_gesture = self.map_gesture_name_to_list_results[gesture_name]
        correct_scores = [result.score for result in results_for_gesture if result.is_correct]
        return correct_scores
    
    def get_incorrect_match_scores_for_gesture(self, gesture_name):
        '''Returns a list of scores for the incorrect matches for this gesture'''
        results_for_gesture = self.map_gesture_name_to_list_results[gesture_name]
        incorrect_scores = [result.score for result in results_for_gesture if not result.is_correct]
        return incorrect_scores
    
    def get_correct_match_indices_in_nbestlist_for_gesture(self, gesture_name):
        '''Returns a list of correct match indices in the n-best list for the given gesture'''
        results = self.map_gesture_name_to_list_results[gesture_name]
        correct_match_indices = [result.get_correct_match_index_nbestlist() for result in results]
        return correct_match_indices
    
    def get_confusion_matrix(self):
        '''
        Returns a scikit learn confusion matrix
        See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        '''
        y_true = list()
        y_pred = list()
        for gesture_name, list_results in self.map_gesture_name_to_list_results.items():
            for result in list_results:
                y_true.append(result.test_trial.gesture_name)
                y_pred.append(result.closest_trial.gesture_name)

        cm_classes = self.get_gesture_names()
        cm = confusion_matrix(y_true, y_pred, labels=cm_classes)
        return cm
    
    def get_nbestlist_performance(self, normalized=True):
        '''Returns a list of accuracies as a function of n-best list position'''
        
        # track the n-best list position to correctness
        map_nbestlist_pos_to_correctness_cnt = dict()
        for gesture_name, results in self.map_gesture_name_to_list_results.items():
            for trial_classification_result in results:
                for i in range(0, len(trial_classification_result.n_best_list_sorted)):
                    matched_trial, score = trial_classification_result.n_best_list_sorted[i]
                    if trial_classification_result.test_trial.gesture_name == matched_trial.gesture_name:
                        # found a correct match
                        if i not in map_nbestlist_pos_to_correctness_cnt:
                            map_nbestlist_pos_to_correctness_cnt[i] = 0 # initialize to zero

                        # at this n-best list position, up the count
                        map_nbestlist_pos_to_correctness_cnt[i] = map_nbestlist_pos_to_correctness_cnt[i] + 1
                        break

        sortedIndices = sorted(map_nbestlist_pos_to_correctness_cnt.keys())

        # now create the n-best list, which will store the number of correct matches
        # at each location in the n-best list
        n_best_list_performance = list()
        cur_value = map_nbestlist_pos_to_correctness_cnt[sortedIndices[0]]
        n_best_list_performance.append(cur_value)
        
        j = 1
        for i in range (1, sortedIndices[-1] + 1):
            if i >= sortedIndices[j]:
                cur_value = cur_value + map_nbestlist_pos_to_correctness_cnt[sortedIndices[j]]
                j = j + 1
            n_best_list_performance.append(cur_value)

        n_best_list_performance = np.array(n_best_list_performance)
        if normalized:
            # print("n_best_list_performance.max(): " + str(n_best_list_performance.max()))
            return n_best_list_performance / n_best_list_performance.max()
        else:
            return n_best_list_performance
        
    
    def print_result(self):
        '''
        Utility function to print results
        '''
        print()
        print("Title:", self.get_title())
        print("Optional arguments:", self.kwargs)
        print("Took {:0.3f}s for {} comparisons (avg={:0.3f}s per match)"
              .format(self.total_time, self.total_num_comparisons, self.get_avg_time_per_comparison()))
        
        for gesture_name in self.get_gesture_names():
            correctness_cnt_for_gesture = self.get_num_correct_for_gesture(gesture_name)
            num_comparisons_for_gesture = len(self.map_gesture_name_to_list_results[gesture_name])
            print("- {} {}/{} ({}%)".format(gesture_name, correctness_cnt_for_gesture, num_comparisons_for_gesture, 
                                            correctness_cnt_for_gesture/num_comparisons_for_gesture * 100))
        
        print(self.get_confusion_matrix())
    
class TrialClassificationResult:
    '''
    Data structure to store the results of a single trial's classification result
    This is the object returned by the find_closest_match_alg functions
    '''
    
    def __init__(self, test_trial, n_best_list_tuple):
        '''
        Parameters:
        test_trial: the test trial
        n_best_list_tuple: a list of tuples where each tuple is (template_trial, score)
        '''
        self.test_trial = test_trial

        # sort the n_best_list by score
        n_best_list_tuple.sort(key=lambda x: x[1])

        self.n_best_list_sorted = n_best_list_tuple
        self.closest_trial = self.n_best_list_sorted[0][0]
        self.score = self.n_best_list_sorted[0][1]
        self.is_correct = test_trial.get_ground_truth_gesture_name() == self.closest_trial.gesture_name
        
        self.fold_idx = -1
        self.elapsed_time = -1 # elapsed time in seconds
        
    def get_correct_match_index_nbestlist(self):
        '''
        Returns the index of the correct match in the n-best list
        '''
        index = 0
        for fold_trial, score in self.n_best_list_sorted:
            if self.test_trial.gesture_name == fold_trial.gesture_name:
                return index
            index += 1    
        return -1
        
    def __str__(self):
        correctness_str = "Correct" if self.is_correct else "Incorrect"
        return("{} : Best match for '{}' Trial {} is '{}' Trial {} w/score: {:0.1f} ({:0.3f}s)".format(
                     correctness_str, self.test_trial.get_ground_truth_gesture_name(), self.test_trial.trial_num, 
                     self.closest_trial.gesture_name, self.closest_trial.trial_num, 
                     self.score, self.elapsed_time))