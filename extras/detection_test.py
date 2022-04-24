########################################
# Raha: The Error Detection System
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# April 2018
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import os
import re
import sys
import math
import time
import json
import random
import pickle
import hashlib
import tempfile
import itertools
import multiprocessing

import numpy
import pandas
import scipy.stats
import scipy.spatial
import scipy.cluster
import sklearn.svm
import sklearn.tree
import sklearn.cluster
import sklearn.ensemble
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.kernel_ridge
import sklearn.neural_network
import sklearn.feature_extraction

import raha
########################################


########################################
class Detection:
    """
    The main class.
    """

    def __init__(self):
        """
        The constructor.
        """
        self.LABELING_BUDGET = 20
        self.USER_LABELING_ACCURACY = 1.0
        self.VERBOSE = False
        self.SAVE_RESULTS = True
        self.CLUSTERING_BASED_SAMPLING = True
        self.STRATEGY_FILTERING = False
        self.CLASSIFICATION_MODEL = "GBC"  # ["ABC", "DTC", "GBC", "GNB", "SGDC", "SVC"]
        self.LABEL_PROPAGATION_METHOD = "homogeneity"   # ["homogeneity", "majority"]
        self.ERROR_DETECTION_ALGORITHMS = ["OD", "PVD", "RVD", "KBVD"]   # ["OD", "PVD", "RVD", "KBVD", "TFIDF"]
        self.HISTORICAL_DATASETS = []
        #selmade
        self.COUNT_CLUSTER = []
        self.COMPARE_SIMILARITY = "matching"
        self.COMPARE_DISTANCE = "euclidean"
        self.COMPARE_MODE = "distance"
        self.DISCOUNT = 1
        self.SUB_CLUSTER_SIZE = 20

    def _strategy_runner_process(self, args):
        """
        This method runs an error detection strategy in a parallel process.
        """
        d, algorithm, configuration = args
        start_time = time.time()
        strategy_name = json.dumps([algorithm, configuration])
        strategy_name_hash = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16))
        outputted_cells = {}
        if algorithm == "OD":
            dataset_path = os.path.join(tempfile.gettempdir(), d.name + "-" + strategy_name_hash + ".csv")
            d.write_csv_dataset(dataset_path, d.dataframe)
            params = ["-F", ",", "--statistical", "0.5"] + ["--" + configuration[0]] + configuration[1:] + [dataset_path]
            raha.tools.dBoost.dboost.imported_dboost.run(params)
            algorithm_results_path = dataset_path + "-dboost_output.csv"
            if os.path.exists(algorithm_results_path):
                ocdf = pandas.read_csv(algorithm_results_path, sep=",", header=None, encoding="utf-8", dtype=str,
                                       keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())
                for i, j in ocdf.values.tolist():
                    if int(i) > 0:
                        outputted_cells[(int(i) - 1, int(j))] = ""
                os.remove(algorithm_results_path)
            try:
                os.remove(dataset_path)
            except:
                print("Still used")
        elif algorithm == "PVD":
            attribute, ch = configuration
            j = d.dataframe.columns.get_loc(attribute)
            for i, value in d.dataframe[attribute].iteritems():
                try:
                    if len(re.findall("[" + ch + "]", value, re.UNICODE)) > 0:
                        outputted_cells[(i, j)] = ""
                except:
                    continue
        elif algorithm == "RVD":
            l_attribute, r_attribute = configuration
            l_j = d.dataframe.columns.get_loc(l_attribute)
            r_j = d.dataframe.columns.get_loc(r_attribute)
            value_dictionary = {}
            for i, row in d.dataframe.iterrows():
                if row[l_attribute]:
                    if row[l_attribute] not in value_dictionary:
                        value_dictionary[row[l_attribute]] = {}
                    if row[r_attribute]:
                        value_dictionary[row[l_attribute]][row[r_attribute]] = 1
            for i, row in d.dataframe.iterrows():
                if row[l_attribute] in value_dictionary and len(value_dictionary[row[l_attribute]]) > 1:
                    outputted_cells[(i, l_j)] = ""
                    outputted_cells[(i, r_j)] = ""
        elif algorithm == "KBVD":
            outputted_cells = raha.tools.KATARA.katara.run(d, configuration)
        detected_cells_list = list(outputted_cells.keys())
        strategy_profile = {
            "name": strategy_name,
            "output": detected_cells_list,
            "runtime": time.time() - start_time
        }
        if self.SAVE_RESULTS:
            pickle.dump(strategy_profile, open(os.path.join(d.results_folder, "strategy-profiling",
                                                            strategy_name_hash + ".dictionary"), "wb"))
        if self.VERBOSE:
            print("{} cells are detected by {}.".format(len(detected_cells_list), strategy_name))
        return strategy_profile

    def initialize_dataset(self, dd):
        """
        This method instantiates the dataset.
        """
        d = raha.dataset.Dataset(dd)
        d.dictionary = dd
        d.results_folder = os.path.join(os.path.dirname(dd["path"]), "raha-baran-results-" + d.name)
        if self.SAVE_RESULTS and not os.path.exists(d.results_folder):
            os.mkdir(d.results_folder)
        d.labeled_tuples = {} if not hasattr(d, "labeled_tuples") else d.labeled_tuples
        d.labeled_cells = {} if not hasattr(d, "labeled_cells") else d.labeled_cells
        d.labels_per_cluster = {} if not hasattr(d, "labels_per_cluster") else d.labels_per_cluster
        d.detected_cells = {} if not hasattr(d, "detected_cells") else d.detected_cells
        #d.labeled_cells_j_c = {} if not hasattr(d, "labeled_cells_j_c") else d.labeled_cells_j_c
        return d

    def run_strategies(self, d):
        """
        This method runs (all or the promising) strategies.
        """
        sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
        if not self.STRATEGY_FILTERING:
            if os.path.exists(sp_folder_path):
                sys.stderr.write("I just load strategies' results as they have already been run on the dataset!\n")
                strategy_profiles_list = [pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                                          for strategy_file in os.listdir(sp_folder_path)]
            else:
                if self.SAVE_RESULTS:
                    os.mkdir(sp_folder_path)
                algorithm_and_configurations = []
                for algorithm_name in self.ERROR_DETECTION_ALGORITHMS:
                    if algorithm_name == "OD":
                        configuration_list = [
                            list(a) for a in
                            list(itertools.product(["histogram"], ["0.1", "0.3", "0.5", "0.7", "0.9"],
                                                   ["0.1", "0.3", "0.5", "0.7", "0.9"])) +
                            list(itertools.product(["gaussian"],
                                                   ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"]))]
                        algorithm_and_configurations.extend(
                            [[d, algorithm_name, configuration] for configuration in configuration_list])
                    elif algorithm_name == "PVD":
                        configuration_list = []
                        for attribute in d.dataframe.columns:
                            column_data = "".join(d.dataframe[attribute].tolist())
                            characters_dictionary = {ch: 1 for ch in column_data}
                            for ch in characters_dictionary:
                                configuration_list.append([attribute, ch])
                        algorithm_and_configurations.extend(
                            [[d, algorithm_name, configuration] for configuration in configuration_list])
                    elif algorithm_name == "RVD":
                        al = d.dataframe.columns.tolist()
                        configuration_list = [[a, b] for (a, b) in itertools.product(al, al) if a != b]
                        algorithm_and_configurations.extend(
                            [[d, algorithm_name, configuration] for configuration in configuration_list])
                    elif algorithm_name == "KBVD":
                        configuration_list = [
                            os.path.join(os.path.dirname(__file__), "tools", "KATARA", "knowledge-base", pat)
                            for pat in os.listdir(os.path.join(os.path.dirname(__file__), "tools", "KATARA", "knowledge-base"))]
                        algorithm_and_configurations.extend(
                            [[d, algorithm_name, configuration] for configuration in configuration_list])
                random.shuffle(algorithm_and_configurations)
                pool = multiprocessing.Pool()
                strategy_profiles_list = pool.map(self._strategy_runner_process, algorithm_and_configurations)
                # pool.close()
                # pool.join()
        else:
            for dd in self.HISTORICAL_DATASETS + [d.dictionary]:
                raha.utilities.dataset_profiler(dd)
                raha.utilities.evaluation_profiler(dd)
            strategy_profiles_list = raha.utilities.get_selected_strategies_via_historical_data(d.dictionary, self.HISTORICAL_DATASETS)
        d.strategy_profiles = strategy_profiles_list
        if self.VERBOSE:
            print("{} strategy profiles are collected.".format(len(d.strategy_profiles)))

    def generate_features(self, d):
        """
        This method generates features.
        """
        columns_features_list = []
        for j in range(d.dataframe.shape[1]):
            feature_vectors = numpy.zeros((d.dataframe.shape[0], len(d.strategy_profiles)))
            for strategy_index, strategy_profile in enumerate(d.strategy_profiles):
                strategy_name = json.loads(strategy_profile["name"])[0]
                if strategy_name in self.ERROR_DETECTION_ALGORITHMS:
                    for cell in strategy_profile["output"]:
                        if cell[1] == j:
                            feature_vectors[cell[0], strategy_index] = 1.0
            if "TFIDF" in self.ERROR_DETECTION_ALGORITHMS:
                vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
                corpus = d.dataframe.iloc[:, j]
                try:
                    tfidf_features = vectorizer.fit_transform(corpus)
                    feature_vectors = numpy.column_stack((feature_vectors, numpy.array(tfidf_features.todense())))
                except:
                    pass
            non_identical_columns = numpy.any(feature_vectors != feature_vectors[0, :], axis=0)
            feature_vectors = feature_vectors[:, non_identical_columns]
            if self.VERBOSE:
                print("{} Features are generated for column {}.".format(feature_vectors.shape[1], j))
            columns_features_list.append(feature_vectors)
        d.column_features = columns_features_list

    def build_clusters(self, d):
        """
        This method builds clusters.
        """
        clustering_results = []
        for j in range(d.dataframe.shape[1]):
            feature_vectors = d.column_features[j]
            clusters_k_c_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
            cells_clusters_k_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
            try:
                clustering_model = scipy.cluster.hierarchy.linkage(feature_vectors, method="average", metric="cosine")
                for k in clusters_k_c_ce:
                    model_labels = [l - 1 for l in
                                    scipy.cluster.hierarchy.fcluster(clustering_model, k, criterion="maxclust")]
                    for index, c in enumerate(model_labels):
                        if c not in clusters_k_c_ce[k]:
                            clusters_k_c_ce[k][c] = {}
                        cell = (index, j)
                        clusters_k_c_ce[k][c][cell] = 1
                        cells_clusters_k_ce[k][cell] = c
            except:
                pass
            if self.VERBOSE:
                print("A hierarchical clustering model is built for column {}.".format(j))
            clustering_results.append([clusters_k_c_ce, cells_clusters_k_ce])
        d.clusters_k_j_c_ce = {k: {j: clustering_results[j][0][k] for j in range(d.dataframe.shape[1])} for k in
                               range(2, self.LABELING_BUDGET + 2)}
        d.cells_clusters_k_j_ce = {k: {j: clustering_results[j][1][k] for j in range(d.dataframe.shape[1])} for k in
                                   range(2, self.LABELING_BUDGET + 2)}

    def sample_tuple(self, d):
        """
        This method samples a tuple.
        """
        # --------------------Calculating Number of Labels per Clusters--------------------
        k = len(d.labeled_tuples) + 2
        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                d.labels_per_cluster[(j, c)] = {cell: d.labeled_cells[cell][0] for cell in d.clusters_k_j_c_ce[k][j][c] if
                                                cell[0] in d.labeled_tuples}
        # --------------------Sampling a Tuple--------------------
        if self.CLUSTERING_BASED_SAMPLING:
            tuple_score = numpy.zeros(d.dataframe.shape[0])
            for i in range(d.dataframe.shape[0]):
                if i not in d.labeled_tuples:
                    score = 0.0
                    for j in range(d.dataframe.shape[1]):
                        if d.clusters_k_j_c_ce[k][j]:
                            cell = (i, j)
                            c = d.cells_clusters_k_j_ce[k][j][cell]
                            score += math.exp(-len(d.labels_per_cluster[(j, c)]))
                    tuple_score[i] = math.exp(score)
        else:
            tuple_score = numpy.ones(d.dataframe.shape[0])
        sum_tuple_score = sum(tuple_score)
        p_tuple_score = tuple_score / sum_tuple_score
        d.sampled_tuple = numpy.random.choice(numpy.arange(d.dataframe.shape[0]), 1, p=p_tuple_score)[0]
        if self.VERBOSE:
            print("Tuple {} is sampled.".format(d.sampled_tuple))

    def label_with_ground_truth(self, d):
        """
        This method labels a tuple with ground truth.
        """
        k = len(d.labeled_tuples) + 2
        d.labeled_tuples[d.sampled_tuple] = 1
        actual_errors_dictionary = d.get_actual_errors_dictionary()
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            user_label = int(cell in actual_errors_dictionary)
            if random.random() > self.USER_LABELING_ACCURACY:
                user_label = 1 - user_label
            d.labeled_cells[cell] = [user_label, d.clean_dataframe.iloc[cell]]
        if self.VERBOSE:
            print("Tuple {} is labeled.".format(d.sampled_tuple))

    def propagate_labels(self, d):
        """
        This method propagates labels.
        """
        d.extended_labeled_cells = {cell: d.labeled_cells[cell][0] for cell in d.labeled_cells}
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            if cell in d.cells_clusters_k_j_ce[k][j]:
                c = d.cells_clusters_k_j_ce[k][j][cell]
                d.labels_per_cluster[(j, c)][cell] = d.labeled_cells[cell][0]
        if self.CLUSTERING_BASED_SAMPLING:
            for j in d.clusters_k_j_c_ce[k]:
                for c in d.clusters_k_j_c_ce[k][j]:
                    if len(d.labels_per_cluster[(j, c)]) > 0:
                        if self.LABEL_PROPAGATION_METHOD == "homogeneity":
                            cluster_label = list(d.labels_per_cluster[(j, c)].values())[0]
                            if sum(d.labels_per_cluster[(j, c)].values()) in [0, len(d.labels_per_cluster[(j, c)])]:
                                for cell in d.clusters_k_j_c_ce[k][j][c]:
                                    d.extended_labeled_cells[cell] = cluster_label
                        elif self.LABEL_PROPAGATION_METHOD == "majority":
                            cluster_label = round(
                                sum(d.labels_per_cluster[(j, c)].values()) / len(d.labels_per_cluster[(j, c)]))
                            for cell in d.clusters_k_j_c_ce[k][j][c]:
                                d.extended_labeled_cells[cell] = cluster_label
        if self.VERBOSE:
            print("The number of labeled data cells increased from {} to {}.".format(len(d.labeled_cells), len(d.extended_labeled_cells)))
    
    def get_similarity(self,vec1,vec2,config):
        #collect 2x2 0/0,0/1,1/0,1/1
        tbl = [0]*4
        for i in range(len(vec1)):
            if vec1[i] == 1 and vec2[i] == 1: tbl[0] += 1
            if vec1[i] == 1 and vec2[i] == 0: tbl[1] += 1
            if vec1[i] == 0 and vec2[i] == 1: tbl[2] += 1
            if vec1[i] == 0 and vec2[i] == 0: tbl[3] += 1
        if config == "matching":
            sim_coeff = (tbl[0]+tbl[3])/sum(tbl)
        if config == "jaccard":
            sim_coeff = tbl[0]/(tbl[0]+tbl[1]+tbl[2])
        if config == "dice":
            sim_coeff = 2*tbl[0]/(2*tbl[0]+tbl[1]+tbl[2])
        if config == "antidice":
            sim_coeff = tbl[0]/(tbl[0]+2*tbl[1]+2*tbl[2])
        if config == "sneath":
            sim_coeff = (2*tbl[0]+2*tbl[3])/(2*tbl[0]+tbl[1]+tbl[2]+2*tbl[3])
        if config == "rogers":
            sim_coeff = (tbl[0]+tbl[3])/(tbl[0]+2*tbl[1]+2*tbl[2]+tbl[3])
        return sim_coeff

    def get_distance(self,vec1,vec2,config):    
        if config == "euclidean":
            dist = numpy.linalg.norm(vec1-vec2,ord=2) / numpy.linalg.norm(numpy.ones(len(vec1)))
        if config == "hamming":
            dist = numpy.linalg.norm(vec1-vec2,ord=1) / len(vec1)
        if config == "normal":
            dist = numpy.linalg.norm(vec1-vec2,ord=2)
        return dist
    
    def propagate_weighted_labels(self, d):
        k = len(d.labeled_tuples) + 2 - 1
        d.labeled_cells_j_c = {j: {c: [{},{}] for c in range(k)} for j in range(d.dataframe.shape[1])}
        # index 0 = labeled, 1 = unlabeled
        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                for cell in d.clusters_k_j_c_ce[k][j][c]:
                    if cell in d.labeled_cells:
                        d.labeled_cells_j_c[j][c][0][cell] = d.labeled_cells[cell][0]
                    else:
                        d.labeled_cells_j_c[j][c][1][cell] = (1,1) #weight,label
                #homogene
                if (sum(d.labeled_cells_j_c[j][c][0].values()) in [0,len(d.labeled_cells_j_c[j][c][0].values())]) and len(d.labeled_cells_j_c[j][c][0].values()) != 0:
                    self.COUNT_CLUSTER[0] += 1
                    for cell1 in d.labeled_cells_j_c[j][c][1]:
                        #calculate binary diff / distance
                        if self.COMPARE_MODE == "distance":
                            min_val = 1
                            for cell2 in d.labeled_cells_j_c[j][c][0]:
                                distance = self.get_distance(d.column_features[j][cell1[0]],d.column_features[j][cell2[0]],self.COMPARE_DISTANCE)
                                if min_val > distance: min_val = distance
                            d.labeled_cells_j_c[j][c][1][cell1] = (0.35*(1-min_val)+0.65,d.labeled_cells_j_c[j][c][0][cell2])
                        elif self.COMPARE_MODE == "similarity":
                            max_val = 0
                            for cell2 in d.labeled_cells_j_c[j][c][0]:
                                similarity = self.get_similarity(d.column_features[j][cell1[0]],d.column_features[j][cell2[0]],self.COMPARE_SIMILARITY)
                                if max_val < similarity: max_val = similarity
                            d.labeled_cells_j_c[j][c][1][cell1] = (0.35*max_val+0.65,d.labeled_cells_j_c[j][c][0][cell2])
                        else:
                            d.labeled_cells_j_c[j][c][1][cell1] = (1,d.labeled_cells_j_c[j][c][0].values()[0])
                           
    def random_walk(self,trans,iterations,label,runs=100):
        start_values = [x for x in range(len(trans)) if label[x] == 1 and sum(trans[x]) != 0]
        if len(start_values) == 0: return label
        all_ways = []
        for _ in range(runs):
            step = numpy.random.choice(start_values)
            way = [step]
            for i in range(iterations):
                if sum(trans[step]) == 0: break
                step = numpy.random.choice(numpy.arange(len(trans)),p=trans[step])
                way.append(step)
            all_ways.extend(way)
        count_cell = [0 for i in range(len(trans))]
        for w in all_ways:
            count_cell[w] += 1
        pre_prop = [x/runs for x in count_cell]
        max_val = max(pre_prop)
        prop = [x/max_val for x in pre_prop]
        for x in range(len(prop)):
            if label[x] == 1: prop[x] = 1.0
        return prop

    def propagate_weighted_labels2(self,d):
        k = len(d.labeled_tuples) + 2 - 1
        d.labeled_cells_j_c = {j: {c: [{},{}] for c in range(k)} for j in range(d.dataframe.shape[1])}
        # index 0 = labeled, 1 = unlabeled
        print(d.labeled_tuples)
        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                #print(c,d.clusters_k_j_c_ce[k][j][c])
                for cell in d.clusters_k_j_c_ce[k][j][c]:
                    #print(cell)
                    if cell in d.labeled_cells:
                        d.labeled_cells_j_c[j][c][0][cell] = d.labeled_cells[cell][0]
                    else:
                        d.labeled_cells_j_c[j][c][1][cell] = (1,1) #weight,label to fit weight and y_train to ai
                #homogene
                if (sum(d.labeled_cells_j_c[j][c][0].values()) in [0,len(d.labeled_cells_j_c[j][c][0].values())]) and len(d.labeled_cells_j_c[j][c][0].values()) != 0:
                    distances = [([self.get_distance(d.column_features[j][cell1[0]],d.column_features[j][cell2[0]],"normal") for cell2 in d.clusters_k_j_c_ce[k][j][c]]) for cell1 in d.clusters_k_j_c_ce[k][j][c]]
                    labeled = [0 for i in range(len(d.clusters_k_j_c_ce[k][j][c]))]
                    max_dist = self.get_distance(numpy.ones(len(d.column_features[j][0])),numpy.zeros(len(d.column_features[j][0])),"normal")
                    for x,cell1 in enumerate(d.clusters_k_j_c_ce[k][j][c]):
                        if cell1[0] in d.labeled_tuples: labeled[x] = 1
                        else:
                            for y,cell2 in enumerate(d.clusters_k_j_c_ce[k][j][c]):
                                if distances[x][y] > max_dist/2 or cell2[0] in d.labeled_tuples: distances[x][y] = 0.0
                        sum_dist = sum(distances[x])
                        if sum_dist != 0.0: distances[x]  = [y/sum_dist for y in distances[x]]
                    cluster_label = sum(d.labeled_cells_j_c[j][c][0].values())
                    print("Start Walking...",len(distances))
                    propabilities = self.random_walk(distances,len(distances),labeled)
                    for x,cell in enumerate(d.clusters_k_j_c_ce[k][j][c]):
                        if labeled[x] == 0: d.labeled_cells_j_c[j][c][1][cell] = (propabilities[x],cluster_label)
                       
    def propagate_weighted_labels3(self,d,eps,s):
        k = len(d.labeled_tuples) + 2 - 1
        d.labeled_cells_j_c = {j: {c: [{},{}] for c in range(k)} for j in range(d.dataframe.shape[1])}
        # index 0 = labeled, 1 = unlabeled
        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                #print(c,d.clusters_k_j_c_ce[k][j][c])
                for cell in d.clusters_k_j_c_ce[k][j][c]:
                    #print(cell)
                    if cell in d.labeled_cells:
                        d.labeled_cells_j_c[j][c][0][cell] = d.labeled_cells[cell][0]
                    else:
                        d.labeled_cells_j_c[j][c][1][cell] = (1,1) #weight,label to fit weight and y_train to ai
                #homogene
                if (sum(d.labeled_cells_j_c[j][c][0].values()) in [0,len(d.labeled_cells_j_c[j][c][0].values())]) and len(d.labeled_cells_j_c[j][c][0].values()) != 0:
                    x_train = [d.column_features[j][cell[0]] for cell in d.clusters_k_j_c_ce[k][j][c]]
                    if len(x_train) < 2: continue
                    db = sklearn.cluster.DBSCAN(eps=eps,min_samples=len(x_train[0])/s).fit(x_train)
                    for x,cell in enumerate(d.clusters_k_j_c_ce[k][j][c]):
                        if db.labels_[x] == -1 and cell in d.labeled_cells_j_c[j][c][1]: d.labeled_cells_j_c[j][c][1][cell] = (0,sum(d.labeled_cells_j_c[j][c][0].values()))
                        elif cell in d.labeled_cells_j_c[j][c][1]: d.labeled_cells_j_c[j][c][1][cell] = (1.0,sum(d.labeled_cells_j_c[j][c][0].values()))
    
    def propagate_weighted_labels4(self,d):
        #
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            if cell in d.cells_clusters_k_j_ce[k][j]:
                c = d.cells_clusters_k_j_ce[k][j][cell]
                d.labels_per_cluster[(j, c)][cell] = d.labeled_cells[cell][0]
        
        d.labeled_cells_j_c = {j: {c: [{},{}] for c in range(k)} for j in range(d.dataframe.shape[1])}
        # index 0 = labeled, 1 = unlabeled
        for j in range(d.dataframe.shape[1]):
            cluster_types = [0,0] # homogene , heterogene
            for c in d.clusters_k_j_c_ce[k][j]:
                #print(c,d.clusters_k_j_c_ce[k][j][c])
                for cell in d.clusters_k_j_c_ce[k][j][c]:
                    #print(cell)
                    if cell in d.labeled_cells:
                        d.labeled_cells_j_c[j][c][0][cell] = (1,d.labeled_cells[cell][0])
                    else:
                        d.labeled_cells_j_c[j][c][1][cell] = (0,-1) #weight,label ;to fit weight and y_train to ai
                #homogene
                if (sum(d.labels_per_cluster[(j, c)].values()) in [0, len(d.labels_per_cluster[(j, c)])]) and len(d.labels_per_cluster[(j, c)]) != 0:
                    cluster_types[0] += 1
                    label = list(d.labels_per_cluster[(j, c)].values())[0]
                    for cell in d.labeled_cells_j_c[j][c][1]:
                        d.labeled_cells_j_c[j][c][1][cell] = (1,label)
                elif len(d.labels_per_cluster[(j, c)]) != 0: cluster_types[1] += 1
            self.COUNT_CLUSTER.append(cluster_types)
                        
    def propagate_weighted_labels5(self, d):
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            if cell in d.cells_clusters_k_j_ce[k][j]:
                c = d.cells_clusters_k_j_ce[k][j][cell]
                d.labels_per_cluster[(j, c)][cell] = d.labeled_cells[cell][0]
                
        d.labeled_cells_j_c = {j: {c: [{},{}] for c in range(k)} for j in range(d.dataframe.shape[1])}
        # index 0 = labeled, 1 = unlabeled
        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                for cell in d.clusters_k_j_c_ce[k][j][c]:
                    if cell in d.labeled_cells:
                        d.labeled_cells_j_c[j][c][0][cell] = (1,d.labeled_cells[cell][0])
                    else:
                        d.labeled_cells_j_c[j][c][1][cell] = (0,-1) #weight,label
                #homogene
                if (sum(d.labels_per_cluster[(j, c)].values()) in [0, len(d.labels_per_cluster[(j, c)])]) and len(d.labeled_cells_j_c[j][c][0].values()) != 0:
                    for cell1 in d.labeled_cells_j_c[j][c][1]:
                        #calculate binary diff / distance
                        if self.COMPARE_MODE == "distance":
                            min_val = 1
                            for cell2 in d.labeled_cells_j_c[j][c][0]:
                                distance = self.get_distance(d.column_features[j][cell1[0]],d.column_features[j][cell2[0]],self.COMPARE_DISTANCE)
                                if min_val > distance: min_val = distance
                            d.labeled_cells_j_c[j][c][1][cell1] = (1-min_val,d.labeled_cells_j_c[j][c][0][cell2][1])
                        elif self.COMPARE_MODE == "similarity":
                            max_val = 0
                            for cell2 in d.labeled_cells_j_c[j][c][0]:
                                similarity = self.get_similarity(d.column_features[j][cell1[0]],d.column_features[j][cell2[0]],self.COMPARE_SIMILARITY)
                                if max_val < similarity: max_val = similarity
                            d.labeled_cells_j_c[j][c][1][cell1] = (max_val,d.labeled_cells_j_c[j][c][0][cell2][1])
                        else:
                            d.labeled_cells_j_c[j][c][1][cell1] = (1,d.labeled_cells_j_c[j][c][0].values()[0])
                        
    def propagate_weighted_labels6(self,d):
        #
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            if cell in d.cells_clusters_k_j_ce[k][j]:
                c = d.cells_clusters_k_j_ce[k][j][cell]
                d.labels_per_cluster[(j, c)][cell] = d.labeled_cells[cell][0]
        
        d.labeled_cells_j_c = {j: {c: [{},{}] for c in range(k)} for j in range(d.dataframe.shape[1])}
        # index 0 = labeled, 1 = unlabeled
        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                #print(c,d.clusters_k_j_c_ce[k][j][c])
                for cell in d.clusters_k_j_c_ce[k][j][c]:
                    #print(cell)
                    if cell in d.labeled_cells:
                        d.labeled_cells_j_c[j][c][0][cell] = (1,d.labeled_cells[cell][0])
                    else:
                        d.labeled_cells_j_c[j][c][1][cell] = (0,-1) #weight,label ;to fit weight and y_train to ai
                        
    def propagate_weighted_labels7(self,d):
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            if cell in d.cells_clusters_k_j_ce[k][j]:
                c = d.cells_clusters_k_j_ce[k][j][cell]
                d.labels_per_cluster[(j, c)][cell] = d.labeled_cells[cell][0]

        d.labeled_cells_j_c = {j: {c: [{},{}] for c in range(k)} for j in range(d.dataframe.shape[1])}
        # index 0 = labeled, 1 = unlabeled
        for j in range(d.dataframe.shape[1]):
            cluster_types = [0,0]
            for c in d.clusters_k_j_c_ce[k][j]:
                #print(c,d.clusters_k_j_c_ce[k][j][c])
                for cell in d.clusters_k_j_c_ce[k][j][c]:
                    #print(cell)
                    if cell in d.labeled_cells:
                        d.labeled_cells_j_c[j][c][0][cell] = (1,d.labeled_cells[cell][0])
                    else:
                        d.labeled_cells_j_c[j][c][1][cell] = (0,-1)
                if (sum(d.labels_per_cluster[(j, c)].values()) in [0, len(d.labels_per_cluster[(j, c)])]) and len(d.labels_per_cluster[(j, c)]) != 0:
                    cluster_types[0] += 1
                    x = [d.column_features[j][i] for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    cells = [(i,j) for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    if len(x) > 1:
                        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=2).fit(x)
                        distances, indices = nbrs.kneighbors(x)
                        distances = numpy.sort(distances, axis=0)[:,1]
                        #calc if slope is over 1% between two distances then choose value as eps
                        eps = distances[0]
                        v1 = 0
                        for x1 in range(len(distances)-1):
                            v2 = distances[x1+1]-distances[x1]
                            if 100*v2 > 101*v1:
                                eps = distances[x1+1]
                                break
                            v1 = v2
                        label = list(d.labels_per_cluster[(j, c)].values())[0]
                        print("Eps:",eps)
                        if eps == 0:
                            #all feature vectors are the same
                            for cell in cells:
                                if cell in d.labeled_cells_j_c[j][c][1]: d.labeled_cells_j_c[j][c][1][cell] = (1,label)
                        else:
                            db = sklearn.cluster.DBSCAN(eps=eps).fit(x)
                            sub_cluster = []
                            for i,cell in enumerate(cells):
                                if cell in d.labeled_cells_j_c[j][c][0] and db.labels_[i] != -1:
                                    sub_cluster.append(db.labels_[i])
                            for i,cell in enumerate(cells):
                                if db.labels_[i] in sub_cluster and cell in d.labeled_cells_j_c[j][c][1]:
                                    d.labeled_cells_j_c[j][c][1][cell] = (1,label)
                elif len(d.labels_per_cluster[(j, c)]) != 0: cluster_types[1] += 1
            self.COUNT_CLUSTER.append(cluster_types)
            
    def propagate_weighted_labels8(self,d):
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            if cell in d.cells_clusters_k_j_ce[k][j]:
                c = d.cells_clusters_k_j_ce[k][j][cell]
                d.labels_per_cluster[(j, c)][cell] = d.labeled_cells[cell][0]

        d.labeled_cells_j_c = {j: {c: [{},{}] for c in range(k)} for j in range(d.dataframe.shape[1])}
        # index 0 = labeled, 1 = unlabeled
        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                #print(c,d.clusters_k_j_c_ce[k][j][c])
                for cell in d.clusters_k_j_c_ce[k][j][c]:
                    #print(cell)
                    if cell in d.labeled_cells:
                        d.labeled_cells_j_c[j][c][0][cell] = (1,d.labeled_cells[cell][0])
                    else:
                        d.labeled_cells_j_c[j][c][1][cell] = (0,-1)
                if (sum(d.labels_per_cluster[(j, c)].values()) in [0, len(d.labels_per_cluster[(j, c)])]) and len(d.labels_per_cluster[(j, c)]) != 0:
                    x = [d.column_features[j][i] for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    cells = [(i,j) for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    if len(x) > 1:
                        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=2).fit(x)
                        distances, indices = nbrs.kneighbors(x)
                        distances = numpy.sort(distances, axis=0)[:,1]
                        #calc if slope is over 1% between two distances then choose value as eps
                        eps = distances[0]
                        v1 = 0
                        for x1 in range(len(distances)-1):
                            v2 = distances[x1+1]-distances[x1]
                            if 100*v2 > 101*v1:
                                eps = distances[x1+1]
                                break
                            v1 = v2
                        label = list(d.labels_per_cluster[(j, c)].values())[0]
                        if eps == 0:
                            #all feature vectors are the same
                            for cell in cells:
                                if cell in d.labeled_cells_j_c[j][c][1]: d.labeled_cells_j_c[j][c][1][cell] = (1.0,label)
                        else:
                            db = sklearn.cluster.DBSCAN(eps=eps).fit(x)
                            sub_cluster = []
                            for i,cell in enumerate(cells):
                                if cell in d.labeled_cells_j_c[j][c][0] and db.labels_[i] != -1:
                                    sub_cluster.append(db.labels_[i])
                            for i,cell in enumerate(cells):
                                if db.labels_[i] in sub_cluster and cell in d.labeled_cells_j_c[j][c][1]:
                                    x_cell = x[cells.index(cell)]
                                    if self.COMPARE_MODE == "distance":
                                        min_dist = 1
                                        for cell2 in d.labeled_cells_j_c[j][c][0]:
                                            x_cell2 = x[cells.index(cell2)]
                                            dist = self.get_distance(x_cell,x_cell2,self.COMPARE_DISTANCE)
                                            if dist < min_dist: min_dist = dist
                                        d.labeled_cells_j_c[j][c][1][cell] = (1-min_dist,label)
                                    elif self.COMPARE_SIMILARITY == "similarity":
                                        max_sim = 0
                                        for cell in d.labeled_cells_j_c[j][c][0]:
                                            x_cell2 = x[cells.index(cell2)]
                                            sim = self.get_similarity(x_cell,x_cell2,self.COMPARE_SIMILARITY)
                                            if sim > max_sim: max_sim = sim
                                        d.labeled_cells_j_c[j][c][1][cell] = (max_dist,label)
    def propagate_weighted_heterogene_labels(self,d):
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            for c in d.labeled_cells_j_c[j]:
                if (sum(d.labeled_cells_j_c[j][c][0].values()) not in [0,len(d.labeled_cells_j_c[j][c][0].values())]) and len(d.labeled_cells_j_c[j][c][0].values()) != 0:
                    self.COUNT_CLUSTER[1] += 1
                    #heterogene
                    sub_cluster_train = [d.column_features[j][i] for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    sub_cluster_cells = [(i,j) for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    #test cluster size to get homogene sub cluster
                    for c_size in range(self.SUB_CLUSTER_SIZE-2):
                        homogene = 0
                        km = sklearn.cluster.KMeans(n_clusters=(c_size+2)).fit(sub_cluster_train)
                        sub_cluster = {sub_c: [{},{}] for sub_c in range(c_size+2)}
                        #sub cluster index 0 = labeled , 1 = unlabeled 
                        for i,cell in enumerate(sub_cluster_cells):
                            if cell in d.labeled_cells_j_c[j][c][0]:
                                sub_cluster[km.labels_[i]][0][cell] = d.labeled_cells_j_c[j][c][0][cell]
                            else:
                                sub_cluster[km.labels_[i]][1][cell] = d.labeled_cells_j_c[j][c][1][cell]
                        #check of sub cluster labels are homogene (counts empty too)
                        for sub_c in sub_cluster:
                            if sum(sub_cluster[sub_c][0].values()) in [0,len(sub_cluster[sub_c][0].values())]:
                                homogene += 1
                        if homogene == len(sub_cluster):
                            #calc weights
                            for sub_c in sub_cluster:
                                if len(sub_cluster[sub_c][0]) != 0:
                                    for cell1 in sub_cluster[sub_c][1]:
                                        if self.COMPARE_MODE == "distance":
                                            min_val = 1
                                            for cell2 in sub_cluster[sub_c][0]:
                                                distance = self.get_distance(d.column_features[j][cell1[0]],d.column_features[j][cell2[0]],self.COMPARE_DISTANCE)
                                                if min_val > distance: min_val = distance
                                            sub_cluster[sub_c][1][cell1] = ((1-distance)*self.DISCOUNT,sub_cluster[sub_c][0][cell2])
                                        if self.COMPARE_MODE == "similarity":
                                            max_val = 0
                                            for cell2 in sub_cluster[sub_c][0]:
                                                similarity = self.get_similarity(d.column_features[j][cell1[0]],d.column_features[j][cell2[0]],self.COMPARE_SIMILARITY)
                                                if max_val < similarity: min_val = similarity
                                            sub_cluster[sub_c][1][cell1] = (similarity*self.DISCOUNT,sub_cluster[sub_c][0][cell2])
                            break
                    for sub_c in sub_cluster:
                        for cell in sub_cluster[sub_c][1]:
                            d.labeled_cells_j_c[j][c][1][cell] = sub_cluster[sub_c][1][cell]
                    
    def propagate_weighted_heterogene_labels2(self,d,e):
        for j in range(d.dataframe.shape[1]):
            for c in d.labeled_cells_j_c[j]:
                if sum(d.labeled_cells_j_c[j][c][0].values()) not in [0,len(d.labeled_cells_j_c[j][c][0].values())]:
                    self.COUNT_CLUSTER[1] += 1
                    for cell1 in d.labeled_cells_j_c[j][c][1]:
                        if self.COMPARE_MODE == "distance":
                            min_val = (1,1) # (weight,label)
                            for cell2 in d.labeled_cells_j_c[j][c][0]:
                                distance = self.get_distance(d.column_features[j][cell1[0]],d.column_features[j][cell2[0]],self.COMPARE_DISTANCE)
                                if min_val[0] > distance: min_val = (distance,d.labeled_cells_j_c[j][c][0][cell2])
                            d.labeled_cells_j_c[j][c][1][cell1] = ((1-min_val[0])**e,min_val[1])
                        if self.COMPARE_MODE == "similarity":
                            max_val = (0,1)
                            for cell2 in d.labeled_cells_j_c[j][c][0]:
                                similarity = self.get_similarity(d.column_features[j][cell1[0]],d.column_features[j][cell2[0]],self.COMPARE_SIMILARITY)
                                if max_val[0] < similarity: max_val = (similarity**e,d.labeled_cells_j_c[j][c][0][cell2])
                            d.labeled_cells_j_c[j][c][1][cell1] = max_val
                            
    def propagate_weighted_heterogene_labels3(self,d):
        from sklearn.semi_supervised import LabelPropagation
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            for c in d.labeled_cells_j_c[j]:
                if sum(d.labeled_cells_j_c[j][c][0].values()) not in [0,len(d.labeled_cells_j_c[j][c][0].values())]:
                    self.COUNT_CLUSTER[1] += 1
                    #print(d.labeled_cells_j_c[j][c][0])
                    sub_cluster_train = [d.column_features[j][i] for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    sub_cluster_cells = [(i,j) for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    sub_cluster_labels = []
                    for i in range(d.dataframe.shape[0]):
                        if (i,j) in d.labeled_cells_j_c[j][c][0]:
                            sub_cluster_labels.append(d.labeled_cells_j_c[j][c][0][(i,j)])
                        elif (i,j) in d.labeled_cells_j_c[j][c][1]:
                            sub_cluster_labels.append(-1)
                    label_prop = LabelPropagation().fit(sub_cluster_train,sub_cluster_labels)
                    closest_labeled_point = []
                    sub_cluster_labeled_features = [sub_cluster_train[i] for i in range(len(sub_cluster_train)) if sub_cluster_cells[i] in d.labeled_cells_j_c[j][c][0]]
                    for feature1 in sub_cluster_train:
                        min_val = self.get_distance(numpy.zeros(len(sub_cluster_train)),numpy.ones(len(sub_cluster_train)),"normal")
                        for feature2 in sub_cluster_labeled_features:
                            tmp = self.get_distance(feature1,feature2,"normal")
                            if tmp < min_val: min_val = tmp
                        closest_labeled_point.append(min_val)
                    for i in range(len(sub_cluster_train)):
                        if sub_cluster_cells[i] in d.labeled_cells_j_c[j][c][1]: d.labeled_cells_j_c[j][c][1][sub_cluster_cells[i]] = (max(label_prop.label_distributions_[i])*(self.DISCOUNT**closest_labeled_point[i]),label_prop.transduction_[i])
 
    def propagate_weighted_heterogene_labels4(self,d):
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                if (sum(d.labels_per_cluster[(j, c)].values()) not in [0, len(d.labels_per_cluster[(j, c)])]) and len(d.labels_per_cluster[(j, c)]) != 0:
                    print("Heterogene")
                    x = [d.column_features[j][i] for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    cells = [(i,j) for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    for sub_cluster in range(20):
                        km = sklearn.cluster.KMeans(n_clusters=(sub_cluster+2)).fit(x)
                        sub_clusters = {sub_c: [] for sub_c in range(sub_cluster+2)}
                        labeled_sub_clusters = []
                        for i,cell in enumerate(cells):
                            #print(cell,km.labels_[i])
                            sub_clusters[km.labels_[i]].append(cell)
                        homogene = True
                        for sub_c in sub_clusters:
                            sub_labeled = {}
                            for cell in sub_clusters[sub_c]:
                                if cell in d.labeled_cells_j_c[j][c][0]: sub_labeled[cell] = d.labeled_cells_j_c[j][c][0][cell][1]
                            if len(sub_labeled) != 0 and sum(sub_labeled.values()) not in [0,len(sub_labeled.values())]: homogene = False
                        if homogene:
                            sub_cluster_number = set()
                            for cell in d.labeled_cells_j_c[j][c][0]:
                                for sub_c in sub_clusters:
                                    if cell in sub_clusters[sub_c]: sub_cluster_number.add((sub_c,d.labeled_cells_j_c[j][c][0][cell][1]))
                            for n in sub_cluster_number:
                                for cell in sub_clusters[n[0]]:
                                    if cell in d.labeled_cells_j_c[j][c][1]: d.labeled_cells_j_c[j][c][1][cell] = (1,n[1])
                            break
                            
    def propagate_weighted_heterogene_labels5(self,d):
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                if (sum(d.labels_per_cluster[(j, c)].values()) not in [0, len(d.labels_per_cluster[(j, c)])]) and len(d.labels_per_cluster[(j, c)]) != 0:
                    x = [d.column_features[j][i] for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    cells = [(i,j) for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    for sub_cluster in range(20):
                        km = sklearn.cluster.KMeans(n_clusters=(sub_cluster+2)).fit(x)
                        sub_clusters = {sub_c: [] for sub_c in range(sub_cluster+2)}
                        labeled_sub_clusters = []
                        for i,cell in enumerate(cells):
                            #print(cell,km.labels_[i])
                            sub_clusters[km.labels_[i]].append(cell)
                        homogene = True
                        for sub_c in sub_clusters:
                            sub_labeled = {}
                            for cell in sub_clusters[sub_c]:
                                if cell in d.labeled_cells_j_c[j][c][0]: sub_labeled[cell] = d.labeled_cells_j_c[j][c][0][cell][1]
                            if len(sub_labeled) != 0 and sum(sub_labeled.values()) not in [0,len(sub_labeled.values())]: homogene = False
                        if homogene:
                            sub_cluster_number = set()
                            for cell in d.labeled_cells_j_c[j][c][0]:
                                for sub_c in sub_clusters:
                                    if cell in sub_clusters[sub_c]:sub_cluster_number.add((sub_c,d.labeled_cells_j_c[j][c][0][cell][1]))
                            
                            for n in sub_cluster_number:
                                x_tmp = [x[i] for i,cell in enumerate(cells) if cell in sub_clusters[n[0]]]
                                cells_tmp = [cell for cell in cells if cell in sub_clusters[n[0]]]
                                if len(x_tmp) > 1:
                                    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=2).fit(x)
                                    distances, indices = nbrs.kneighbors(x)
                                    distances = numpy.sort(distances, axis=0)[:,1]
                                    #calc if slope is over 1% between two distances then choose value as eps
                                    eps = distances[0]
                                    v1 = 0
                                    for x1 in range(len(distances)-1):
                                        v2 = distances[x1+1]-distances[x1]
                                        if 100*v2 > 101*v1:
                                            eps = distances[x1+1]
                                            break
                                        v1 = v2
                                    label = n[1]
                                    if eps == 0:
                                        for cell in sub_clusters[n[0]]:
                                            if cell in d.labeled_cells_j_c[j][c][1]: d.labeled_cells_j_c[j][c][1][cell] = (1,label)
                                    else:
                                        db = sklearn.cluster.DBSCAN(eps=eps).fit(x_tmp)
                                        db_cluster = []
                                        for i,cell in enumerate(cells_tmp):
                                            if cell in d.labeled_cells_j_c[j][c][0] and db.labels_[i] != -1:
                                                db_cluster.append(db.labels_[i])
                                        for i,cell in enumerate(cells_tmp):
                                            if db.labels_[i] in db_cluster and cell in d.labeled_cells_j_c[j][c][1]:
                                                d.labeled_cells_j_c[j][c][1][cell] = (1,label)
                            break
                            
    def propagate_weighted_heterogene_labels6(self,d):
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                if (sum(d.labels_per_cluster[(j, c)].values()) not in [0, len(d.labels_per_cluster[(j, c)])]) and len(d.labels_per_cluster[(j, c)]) != 0:
                    x = [d.column_features[j][i] for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    cells = [(i,j) for i in range(d.dataframe.shape[0]) if (i,j) in d.clusters_k_j_c_ce[k][j][c]]
                    for sub_cluster in range(20):
                        km = sklearn.cluster.KMeans(n_clusters=(sub_cluster+2)).fit(x)
                        sub_clusters = {sub_c: [] for sub_c in range(sub_cluster+2)}
                        labeled_sub_clusters = []
                        for i,cell in enumerate(cells):
                            #print(cell,km.labels_[i])
                            sub_clusters[km.labels_[i]].append(cell)
                        homogene = True
                        for sub_c in sub_clusters:
                            sub_labeled = {}
                            for cell in sub_clusters[sub_c]:
                                if cell in d.labeled_cells_j_c[j][c][0]: sub_labeled[cell] = d.labeled_cells_j_c[j][c][0][cell][1]
                            if len(sub_labeled) != 0 and sum(sub_labeled.values()) not in [0,len(sub_labeled.values())]: homogene = False
                        if homogene:
                            sub_cluster_number = set()
                            for cell in d.labeled_cells_j_c[j][c][0]:
                                for sub_c in sub_clusters:
                                    if cell in sub_clusters[sub_c]:sub_cluster_number.add((sub_c,d.labeled_cells_j_c[j][c][0][cell][1]))
                            
                            for n in sub_cluster_number:
                                sub_c = n[0]
                                x_tmp = [x[i] for i,cell in enumerate(cells) if cell in sub_clusters[sub_c]]
                                cells_tmp = [cell for cell in cells if cell in sub_clusters[sub_c]]
                                if len(x_tmp) > 1:
                                    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=2).fit(x)
                                    distances, indices = nbrs.kneighbors(x)
                                    distances = numpy.sort(distances, axis=0)[:,1]
                                    #calc if slope is over 1% between two distances then choose value as eps
                                    eps = distances[0]
                                    v1 = 0
                                    for x1 in range(len(distances)-1):
                                        v2 = distances[x1+1]-distances[x1]
                                        if 100*v2 > 101*v1:
                                            eps = distances[x1+1]
                                            break
                                        v1 = v2
                                    label = n[1]
                                    if eps == 0:
                                        for cell in sub_clusters[sub_c]:
                                            if cell in d.labeled_cells_j_c[j][c][1]: d.labeled_cells_j_c[j][c][1][cell] = (1.0,label)
                                    else:
                                        db = sklearn.cluster.DBSCAN(eps=eps).fit(x_tmp)
                                        db_cluster = []
                                        closest_dist = {}
                                        for i,cell in enumerate(cells_tmp):
                                            if cell in d.labeled_cells_j_c[j][c][0] and db.labels_[i] != -1:
                                                db_cluster.append(db.labels_[i])
                                        for i,cell in enumerate(cells_tmp):
                                            if db.labels_[i] in db_cluster and cell in d.labeled_cells_j_c[j][c][1]:
                                                x_cell = x[cells.index(cell)]
                                                if self.COMPARE_MODE == "distance":
                                                    min_dist = 1
                                                    for cell2 in d.labeled_cells_j_c[j][c][0]:
                                                        x_cell2 = x[cells.index(cell2)]
                                                        dist = self.get_distance(x_cell,x_cell2,self.COMPARE_DISTANCE)
                                                        if dist < min_dist: min_dist = dist
                                                    d.labeled_cells_j_c[j][c][1][cell] = (1-min_dist,label)
                                                elif self.COMPARE_MODE == "similarity":
                                                    max_sim = 0
                                                    for cell2 in d.labeled_cells_j_c[j][c]:
                                                        x_cell2 = x[cells.index(cell2)]
                                                        sim = self.get_similarity(x_cell1,x_cell2,self.COMPARE_SIMILARITY)
                                                        if sim > max_sim: max_sim = sim
                                                    d.labeled_cells_j_c[j][c][1][cell] = (max_sim,label)
                            break

    def predict_weighted_labels(self, d):
        #append weight to features
        k = len(d.labeled_tuples) + 2 - 1
        detected_cells_dictionary = {}
        for j in range(d.dataframe.shape[1]):
            x_train = []
            y_train = []
            weights = []
            for c in range(k):
                if len(d.labeled_cells_j_c[j][c][0]) != 0:
                    for x1 in d.labeled_cells_j_c[j][c][0]:
                        x_train.append(d.column_features[j][x1[0]])
                        y_train.append(d.labeled_cells_j_c[j][c][0][x1])
                        weights.append(1.0)
                    if len(d.labeled_cells_j_c[j][c][1]) != 0 and sum(d.labeled_cells_j_c[j][c][0].values()) in [0,len(d.labeled_cells_j_c[j][c][0])]:
                        for x2 in d.labeled_cells_j_c[j][c][1]:
                            x_train.append(d.column_features[j][x2[0]])
                            y_train.append(d.labeled_cells_j_c[j][c][1][x2][1])
                            weights.append(d.labeled_cells_j_c[j][c][1][x2][0])
                        #print(d.labeled_cells_j_c[j][c])
            if sum(y_train) == len(y_train):
                y_pred = numpy.ones(d.dataframe.shape[0])
            elif sum(y_train) == 0 or len(x_train[0]) == 0:
                y_pred = numpy.zeros(d.dataframe.shape[0])
            else:
                if self.CLASSIFICATION_MODEL == "GNB":
                    model = sklearn.naive_bayes.GaussianNB()
                if self.CLASSIFICATION_MODEL == "ABC":
                    model = sklearn.ensemble.AdaBoostClassifier()
                if self.CLASSIFICATION_MODEL == "DTC":
                    model = sklearn.tree.DecisionTreeClassifier()
                if self.CLASSIFICATION_MODEL == "GBC":
                    model = sklearn.ensemble.GradientBoostingClassifier()
                if self.CLASSIFICATION_MODEL == "SGDC":
                    model = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
                if self.CLASSIFICATION_MODEL == "SVC":
                    model = sklearn.svm.SVC(kernel="sigmoid")
                model.fit(x_train,y_train,sample_weight=weights)
                y_pred = model.predict(d.column_features[j])
            for i,y in enumerate(y_pred):
                if (i in d.labeled_tuples and d.labeled_cells[(i,j)][0]) or (i not in d.labeled_tuples and y):
                    detected_cells_dictionary[(i,j)] = "Just a dummy value"
            d.detected_cells.update(detected_cells_dictionary)
            
    def predict_weighted_labels2(self,d):
        #append weight to features
        homogene = True
        k = len(d.labeled_tuples) + 2 - 1
        detected_cells_dictionary = {}
        d.extended_labeled_cells = {cell: (1,d.labeled_cells[cell][0]) for cell in d.labeled_cells}
        for j in range(d.dataframe.shape[1]):
            for c in d.labeled_cells_j_c[j]:
                if len(d.labeled_cells_j_c[j][c][0]) != 0:
                    #d.extended_labeled_cells.update(d.labeled_cells_j_c[j][c][0])
                    if self.LABEL_PROPAGATION_METHOD == "homogeneity" and sum(d.labels_per_cluster[(j, c)].values()) in [0,len(d.labels_per_cluster[(j, c)].values())]:
                        d.extended_labeled_cells.update(d.labeled_cells_j_c[j][c][1])
                    elif self.LABEL_PROPAGATION_METHOD == "heterogeneity":
                        d.extended_labeled_cells.update(d.labeled_cells_j_c[j][c][1])
        #print("Ext len:",len(d.extended_labeled_cells))
        for j in range(d.dataframe.shape[1]):
            feature_vectores = d.column_features[j]
            x_train = [feature_vectores[i] for i in range(d.dataframe.shape[0]) if (i,j) in d.extended_labeled_cells and d.extended_labeled_cells[(i,j)][1] != -1]
            y_train = [d.extended_labeled_cells[(i,j)][1] for i in range(d.dataframe.shape[0]) if (i,j) in d.extended_labeled_cells and d.extended_labeled_cells[(i,j)][1] != -1]
            weights = [d.extended_labeled_cells[(i,j)][0] for i in range(d.dataframe.shape[0]) if (i,j) in d.extended_labeled_cells and d.extended_labeled_cells[(i,j)][1] != -1]
            #print("Len x:",len(x_train))
            if sum(y_train) == len(y_train):
                y_pred = numpy.ones(d.dataframe.shape[0])
            elif sum(y_train) == 0 or len(x_train[0]) == 0:
                y_pred = numpy.zeros(d.dataframe.shape[0])
            else:
                if self.CLASSIFICATION_MODEL == "GNB":
                    model = sklearn.naive_bayes.GaussianNB()
                if self.CLASSIFICATION_MODEL == "ABC":
                    model = sklearn.ensemble.AdaBoostClassifier(n_estimators=50)
                if self.CLASSIFICATION_MODEL == "GBC":
                    model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
                if self.CLASSIFICATION_MODEL == "DTC":
                    model = sklearn.tree.DecisionTreeClassifier(criterion="gini",class_weight={0: len(y_train)/(len(y_train)-sum(y_train)), 1: len(y_train)/sum(y_train)})
                model.fit(x_train,y_train,sample_weight=weights)
                y_pred = model.predict(d.column_features[j])
            for i,y in enumerate(y_pred):
                #print((i,j),y)
                if (i in d.labeled_tuples and d.labeled_cells[(i,j)][0]) or (i not in d.labeled_tuples and y):
                    detected_cells_dictionary[(i,j)] = "Just a dummy value"
            d.detected_cells.update(detected_cells_dictionary)

    def predict_labels(self, d):
        """
        This method predicts the label of data cells.
        """
        detected_cells_dictionary = {}
        #print("Ext Len:",len(d.extended_labeled_cells))
        for j in range(d.dataframe.shape[1]):
            feature_vectors = d.column_features[j]
            x_train = [feature_vectors[i, :] for i in range(d.dataframe.shape[0]) if (i, j) in d.extended_labeled_cells]
            y_train = [d.extended_labeled_cells[(i, j)] for i in range(d.dataframe.shape[0]) if
                       (i, j) in d.extended_labeled_cells]
            x_test = feature_vectors
            #print("Len x:",len(x_train))
            if sum(y_train) == len(y_train):
                predicted_labels = numpy.ones(d.dataframe.shape[0])
            elif sum(y_train) == 0 or len(x_train[0]) == 0:
                predicted_labels = numpy.zeros(d.dataframe.shape[0])
            else:
                if self.CLASSIFICATION_MODEL == "ABC":
                    classification_model = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
                if self.CLASSIFICATION_MODEL == "DTC":
                    classification_model = sklearn.tree.DecisionTreeClassifier(criterion="gini")
                if self.CLASSIFICATION_MODEL == "GBC":
                    classification_model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
                if self.CLASSIFICATION_MODEL == "GNB":
                    classification_model = sklearn.naive_bayes.GaussianNB()
                if self.CLASSIFICATION_MODEL == "KNC":
                    classification_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
                if self.CLASSIFICATION_MODEL == "SGDC":
                    classification_model = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
                if self.CLASSIFICATION_MODEL == "SVC":
                    classification_model = sklearn.svm.SVC(kernel="sigmoid")
                classification_model.fit(x_train, y_train)
                predicted_labels = classification_model.predict(x_test)
            for i, pl in enumerate(predicted_labels):
                if (i in d.labeled_tuples and d.extended_labeled_cells[(i, j)]) or (i not in d.labeled_tuples and pl):
                    detected_cells_dictionary[(i, j)] = "JUST A DUMMY VALUE"
            if self.VERBOSE:
                print("A classifier is trained and applied on column {}.".format(j))
        d.detected_cells.update(detected_cells_dictionary)

    def store_results(self, d):
        """
        This method stores the results.
        """
        ed_folder_path = os.path.join(d.results_folder, "error-detection")
        if not os.path.exists(ed_folder_path):
            os.mkdir(ed_folder_path)
        pickle.dump(d, open(os.path.join(ed_folder_path, "detection.dataset"), "wb"))
        if self.VERBOSE:
            print("The results are stored in {}.".format(os.path.join(ed_folder_path, "detection.dataset")))

    def run(self, dd):
        """
        This method runs Raha on an input dataset to detection data errors.
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------------Initializing the Dataset Object--------------------\n"
                  "------------------------------------------------------------------------")
        d = self.initialize_dataset(dd)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-------------------Running Error Detection Strategies-------------------\n"
                  "------------------------------------------------------------------------")
        self.run_strategies(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-----------------------Generating Feature Vectors-----------------------\n"
                  "------------------------------------------------------------------------")
        self.generate_features(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------Building the Hierarchical Clustering Model---------------\n"
                  "------------------------------------------------------------------------")
        self.build_clusters(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-------------Iterative Clustering-Based Sampling and Labeling-----------\n"
                  "------------------------------------------------------------------------")
        while len(d.labeled_tuples) < self.LABELING_BUDGET:
            self.sample_tuple(d)
            if d.has_ground_truth:
                self.label_with_ground_truth(d)
            # else:
            #   In this case, user should label the tuple interactively as shown in the Jupyter notebook.
            if self.VERBOSE:
                print("------------------------------------------------------------------------")
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------Propagating User Labels Through the Clusters--------------\n"
                  "------------------------------------------------------------------------")
        self.propagate_labels(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------Training and Testing Classification Models---------------\n"
                  "------------------------------------------------------------------------")
        self.predict_labels(d)
        if self.SAVE_RESULTS:
            if self.VERBOSE:
                print("------------------------------------------------------------------------\n"
                      "---------------------------Storing the Results--------------------------\n"
                      "------------------------------------------------------------------------")
            self.store_results(d)
        return d.detected_cells
########################################


########################################
if __name__ == "__main__":
    dataset_name = "flights"
    dataset_dictionary = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
        "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
    }
    app = Detection()
    detection_dictionary = app.run(dataset_dictionary)
    data = raha.dataset.Dataset(dataset_dictionary)
    p, r, f = data.get_data_cleaning_evaluation(detection_dictionary)[:3]
    print("Raha's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name, p, r, f))
    # --------------------
    # app.STRATEGY_FILTERING = True
    # app.HISTORICAL_DATASETS = [
    #     {
    #     "name": "hospital",
    #     "path": "/media/mohammad/C20E45C80E45B5E7/Projects/raha/datasets/hospital/dirty.csv",
    #     "clean_path": "/media/mohammad/C20E45C80E45B5E7/Projects/raha/datasets/hospital/clean.csv"
    #     },
    #     {
    #     "name": "beers",
    #     "path": "/media/mohammad/C20E45C80E45B5E7/Projects/raha/datasets/beers/dirty.csv",
    #     "clean_path": "/media/mohammad/C20E45C80E45B5E7/Projects/raha/datasets/beers/clean.csv"
    #     }
    # ]
    # detection_dictionary = app.run(dataset_dictionary)
    # data = raha.dataset.Dataset(dataset_dictionary)
    # p, r, f = data.get_data_cleaning_evaluation(detection_dictionary)[:3]
    # print("Raha's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name, p, r, f))
#######################################
