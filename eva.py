import raha
import os
import sys
import copy
import numpy
import plotext as plt
import csv
import array

def evaluation(dd,runs,configs,budget):
    stats = [[] for _ in range(len(configs))]
    det = raha.detection.Detection()
    det.LABELING_BUDGET = budget
    d = det.initialize_dataset(dd)
    #run
    det.run_strategies(d)
    det.generate_features(d)
    for run in range(runs):
        print("Run:",run)
        det_tmp = copy.deepcopy(det)
        d_tmp = copy.deepcopy(d)
        det_tmp.build_clusters(d_tmp)
        while len(d_tmp.labeled_tuples) < det_tmp.LABELING_BUDGET:
            det_tmp.sample_tuple(d_tmp)
            if d_tmp.has_ground_truth:
                det_tmp.label_with_ground_truth(d_tmp)
        for i in range(len(configs)):
            det_tmp_tmp = copy.deepcopy(det_tmp)
            d_tmp_tmp = copy.deepcopy(d_tmp)
            print("Configuration:",configs[i])
            if configs[i] == "normal":
                det_tmp_tmp.propagate_labels(d_tmp_tmp)
                det_tmp_tmp.predict_labels(d_tmp_tmp)
            if configs[i] == "prop6":
                det_tmp_tmp.propagate_weighted_labels6(d_tmp_tmp)
                det_tmp_tmp.predict_weighted_labels(d_tmp_tmp)
            if configs[i] == "prop7":
                det_tmp_tmp.propagate_weighted_labels7(d_tmp_tmp)
                det_tmp_tmp.predict_weighted_labels(d_tmp_tmp)
            if configs[i] == "prop8":
                det_tmp_tmp.propagate_weighted_labels8(d_tmp_tmp)
                det_tmp_tmp.predict_weighted_labels(d_tmp_tmp)
            if configs[i] == "prop7_het4":
                det_tmp_tmp.LABEL_PROPAGATION_METHOD = "heterogeneity"
                det_tmp_tmp.propagate_weighted_labels7(d_tmp_tmp)
                det_tmp_tmp.propagate_weighted_heterogene_labels4(d_tmp_tmp)
                det_tmp_tmp.predict_weighted_labels(d_tmp_tmp)
            if configs[i] == "prop7_het5":
                det_tmp_tmp.LABEL_PROPAGATION_METHOD = "heterogeneity"
                det_tmp_tmp.propagate_weighted_labels7(d_tmp_tmp)
                det_tmp_tmp.propagate_weighted_heterogene_labels5(d_tmp_tmp)
                det_tmp_tmp.predict_weighted_labels(d_tmp_tmp)
            if configs[i] == "prop8_het4":
                det_tmp_tmp.LABEL_PROPAGATION_METHOD = "heterogeneity"
                det_tmp_tmp.propagate_weighted_labels8(d_tmp_tmp)
                det_tmp_tmp.propagate_weighted_heterogene_labels4(d_tmp_tmp)
                det_tmp_tmp.predict_weighted_labels(d_tmp_tmp)
            if configs[i] == "prop8_het5":
                det_tmp_tmp.LABEL_PROPAGATION_METHOD = "heterogeneity"
                det_tmp_tmp.propagate_weighted_labels8(d_tmp_tmp)
                det_tmp_tmp.propagate_weighted_heterogene_labels5(d_tmp_tmp)
                det_tmp_tmp.predict_weighted_labels(d_tmp_tmp)
            if configs[i] in ["euclidean","matching","hamming","dice","jacard","dot","cosine"]:
                if configs[i] in ["euclidean","hamming"]:
                    det_tmp_tmp.COMPARE_MODE = "distance"
                    det_tmp_tmp.COMPARE_DISTANCE = configs[i]
                if configs[i] in ["matching","jaccard","dice","sneath","dot","cosine"]:
                    det_tmp_tmp.COMAPRE_MODE = "similarity"
                    det_tmp_tmp.COMPARE_SIMILARITY = configs[i]
                det_tmp_tmp.propagate_weighted_labels_test(d_tmp_tmp)
                det_tmp_tmp.predict_weighted_labels(d_tmp_tmp)
            if configs[i] in ["euclidean_het","matching_het","hamming_het","dice_het","jacard_het","dot_het","cosine_het"]:
                if configs[i] in ["euclidean_het","hamming_het"]:
                    det_tmp_tmp.LABEL_PROPAGATION_METHOD = "heterogeneity"
                    det_tmp_tmp.COMPARE_MODE = "distance"
                    det_tmp_tmp.COMPARE_DISTANCE = configs[i][:len(configs[i])-4]
                if configs[i] in ["matching_het","jaccard_het","dice_het","sneath_het","dot_het","cosine_het"]:
                    det_tmp_tmp.LABEL_PROPAGATION_METHOD = "heterogeneity"
                    det_tmp_tmp.COMAPRE_MODE = "similarity"
                    det_tmp_tmp.COMPARE_SIMILARITY = configs[i][:len(configs[i])-4]
                det_tmp_tmp.propagate_weighted_labels_test(d_tmp_tmp)
                det_tmp_tmp.propagate_weighted_heterogene_labels3(d_tmp_tmp)
                det_tmp_tmp.predict_weighted_labels(d_tmp_tmp)
            if configs[i] == "normal_het":
                det_tmp_tmp.LABEL_PROPAGATION_METHOD = "majority"
                det_tmp_tmp.propagate_labels(d_tmp_tmp)
                det_tmp_tmp.predict_labels(d_tmp_tmp)
            if configs[i] == "marked":
                det_tmp_tmp.propagate_weighted_labels_marked(d_tmp_tmp)
                det_tmp_tmp.predict_weighted_labels(d_tmp_tmp)
                       
            #cleaning evaluation
            data = raha.dataset.Dataset(dd)
            p,r,f = data.get_data_cleaning_evaluation(d_tmp_tmp.detected_cells)[:3]
            stats[i].append(f)
            #print("Raha's performance on {}:\nPrecison = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name,p,r,f))
    return stats

def cluster_test(dd,lb):
    det = raha.detection.Detection()
    det.LABELING_BUDGET = lb
    d = det.initialize_dataset(dd)
    det.run_strategies(d)
    det.generate_features(d)
    det.build_clusters(d)
    while len(d.labeled_tuples) < det.LABELING_BUDGET:
        det.sample_tuple(d)
        if d.has_ground_truth:
            det.label_with_ground_truth(d)
    hom = 0
    het = 0
    k = len(d.labeled_tuples) + 2 - 1
    for j in range(d.dataframe.shape[1]):
        for c in d.clusters_k_j_c_ce[k][j]:
            if len(d.labels_per_cluster[(j,c)].values()) != 0 and sum(d.labels_per_cluster[(j,c)].values()) in [0,len(d.labels_per_cluster[(j,c)])]: hom += 1
            elif len(d.labels_per_cluster[(j,c)].values()) != 0: het += 1
    return hom,het

if __name__ == "__main__":
    args = sys.argv
    print("Args:",args[1:])
    mode = str(args[1])
    if mode == "f":
        runs = int(args[2])
        configs = str(args[3]).split(',')
        all_stats = []
        datasets = []
        for d_name in args[4:]:
            print("Start",d_name)
            datasets.append(d_name)
            dd = {
                    "name": d_name  ,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", d_name, "dirty.csv")) ,
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", d_name, "clean.csv"))
                    }
            all_stats.append(evaluation(dd,runs,configs,20))
        print("Total runs:",runs)
        for i in range(len(datasets)):
            pre_lines = []
            f_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,"history",datasets[i]+"_eva_data.csv"))
            f = open(f_path,"w",encoding="UTF8")
            f_writer = csv.writer(f)
            print("Dataset:",datasets[i])
            for c in range(len(configs)):
                ex = False
                for pre in pre_lines:
                    if pre[0] == configs[c]:
                        pre.extend(all_stats[i][c])
                        ex = True
                        break
                if not ex:
                    data = [configs[c]]
                    data.extend(all_stats[i][c])
                    pre_lines.append(data)
                #print("Configuration:",configs[c],"F1 score:", sum(all_stats[i][c])/runs,"Std:",numpy.std(all_stats[i][c]))
            f_writer.writerows(pre_lines)
            f.close()
    elif mode == "b":
        budget = int(args[2])
        configs = str(args[3]).split(',')
        all_stats = []
        datasets = []
        for d_name in args[4:]:
            print("Start",d_name)
            datasets.append(d_name)
            dd = {
                    "name": d_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", d_name, "dirty.csv")) ,
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", d_name, "clean.csv"))
                    }
            count = budget
            stats_tmp = []
            while count <= 100:
                print("Count",count)
                r = 3 
                stats_tmp.append(evaluation(dd,r,configs,count))
                count += budget

            all_stats.append(stats_tmp)
            #all_stats.append([[[1,1,1],[2,2,2]],[[1,1,1],[2,2,2]],[[1,1,1],[2,2,2]],[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]])

        for i in range(len(datasets)):
            val = [[configs[c]] for c in range(len(configs))]
            for st in all_stats[i]:
                print(st)
                for c in range(len(configs)):
                    val[c].append(st[c])
            print(val)

            for c in range(len(configs)):
                pre_lines = [[] for x in range(101)]
                f_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,"history",datasets[i]+"_"+configs[c]+"_clu_data.csv"))
                f = open(f_path,"w",encoding="UTF8")
                f_writer = csv.writer(f,delimiter=",")
                
                for j in range(len(val[c][1:])):
                    pre_lines[(j+1)*budget].extend(val[c][1:][j])
                for line in pre_lines:
                    print(line)
                    f_writer.writerow(line)
                f.close()
    elif mode == "b1":
        budget = int(args[2])
        configs = ["normal","marked","prop7","prop8","prop7_het4","prop7_het5","prop8_het4","prop8_het5"]
        all_stats = []
        datasets = []
        for d_name in args[3:]:
            print("Start",d_name)
            datasets.append(d_name)
            dd = {
                    "name": d_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", d_name, "dirty.csv")) ,
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", d_name, "clean.csv"))
                    }
            count = budget
            stats_tmp = []
            while count <= 100:
                print("Count",count)
                r = 3
                stats_tmp.append(evaluation(dd,r,configs,count))
                count += budget

            all_stats.append(stats_tmp)
            #all_stats.append([[[1,1,1],[2,2,2]],[[1,1,1],[2,2,2]],[[1,1,1],[2,2,2]],[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]])

        for i in range(len(datasets)):
            val = [[configs[c]] for c in range(len(configs))]
            for st in all_stats[i]:
                print(st)
                for c in range(len(configs)):
                    val[c].append(st[c])
            print(val)

            for c in range(len(configs)):
                pre_lines = [[] for x in range(101)]
                f_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,"history",datasets[i]+"_"+configs[c]+"_clu_data.csv"))
                f = open(f_path,"w",encoding="UTF8")
                f_writer = csv.writer(f,delimiter=",")
                
                for j in range(len(val[c][1:])):
                    pre_lines[(j+1)*budget].extend(val[c][1:][j])
                for line in pre_lines:
                    print(line)
                    f_writer.writerow(line)
                f.close()
    elif mode == "c":
        #ratio between heterogene  clusters and homogene
        budget = int(args[2])
        all_stats = []
        for d_name in args[3:]:
            stats = []
            print("Start",d_name) 
            dd = {
                    "name": d_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", d_name, "dirty.csv")) ,
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", d_name, "clean.csv"))
                    }
            b = budget
            r = 4
            while b <= 100:
                print(b)
                cl = [0,0]
                for _ in range(r):
                    c = cluster_test(dd,b)
                    cl[0] += c[0]
                    cl[1] += c[1]
                stats.append(c[1]/(c[0]+c[1]))
                b += budget
            st = [d_name]
            st.extend(stats)
            all_stats.append(st)

        f_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,"history","hom_het_ration.csv"))
        f = open(f_path,"w",encoding="UTF8")
        f_writer = csv.writer(f)
        for s in all_stats:
            f_writer.writerow(s)
        f.close()
        
    elif mode == "t":
        runs = int(args[2])
        for d_name in args[3:]:
            dd = {
                    "name": d_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", d_name, "dirty.csv")) ,
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", d_name, "clean.csv"))
                    } 
    
            configs = ["normal","normal_het","marked","prop7_het5","euclidean_het"]
            all_stats = evaluation(dd,runs,configs,20)
            print(all_stats)

            pre_lines = []
            f_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,"history",d_name+"_final_test_data.csv"))
            f = open(f_path,"w",encoding="UTF8")
            f_writer = csv.writer(f)

            for i in range(int(len(all_stats)/len(configs))):
                tmp = []
                for c in range(len(configs)):
                    tmp.append(all_stats[len(configs)*i+c])
                pre_lines.append(tmp)
            f_writer.writerows(pre_lines)
            f.close()

        #f = open(os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,"history","_eva_data.csv")),'w')
        #csv.writer(f).writerow([23,23,23,23,23])
        #f.close()




