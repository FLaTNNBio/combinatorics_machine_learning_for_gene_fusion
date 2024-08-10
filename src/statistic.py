import argparse
import pandas as pd
import numpy as np
import argparse

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# FORMAT = INFO1 : value1 - INFO2 : value2 - ........... - INFOk : valuek
def get_metrics(path='fingerprint/test/',file_results_name = 'fingerprint/test/test_pipeline_FILTER_AUTOENCODER_with_fingerprint_no_list_mean.txt', problem = 'classification'):

    file_results = open(path + file_results_name, 'r')
    results = file_results.readlines()

    file_experiment = open('list_experiment.txt')
    list_genes = file_experiment.readlines()
    list_genes = [s.replace('\n', '') for s in list_genes]

    y_pred = []
    y_real = []
    labels = []

    for line in results:
        infos = line.split("-")

        for info in infos:

            info_splitted = info.split()

            # ID PREDICTED
            info_splitted[0] = (info_splitted[0]).replace(':','')
            if info_splitted[0] == "ID_PREDICTED":
                y_pred.append(info_splitted[1])

            # ID LABEL
            if info_splitted[0] == "ID_LABEL":

                label = ''
                if problem == 'classification':
                    y_real.append(info_splitted[1])
                    label = info_splitted[1]
                else:
                    if info_splitted[1] in list_genes:
                        label = info_splitted[1]
                        y_real.append(info_splitted[1])
                    else:
                        label = info_splitted['NONE']
                        y_real.append('NONE')

                labels.append(label)

    clsf = classification_report(y_real, y_pred, output_dict=True)
    clsf_report = pd.DataFrame(data=clsf).transpose()
    name = file_results_name.replace('.txt','')
    clsf_report.to_csv(path + 'clsf_report_' + name + '.csv', index= True)

    labels = np.unique(labels)
    cm = confusion_matrix(y_real, y_pred, labels=labels)
    cm_report = pd.DataFrame(cm, index=labels, columns=labels).transpose()
    cm_report.to_csv(path + 'cm_report_' + name + '.csv', index=True)


# Compute number of conflicts (args.step = 'conflicts') ##############################################################################
def conflicts_statistic_step(args):

    factorization = args.type_factorization
    k = args.k_value
    path = args.path

    dict_conflicts = {}

    input_fingerprint = path + 'fingerprint_' + factorization + '.txt'
    input_fact = path + 'fact_fingerprint_' + factorization + '.txt'

    fingerprint_file = open(input_fingerprint)
    fingerprints = fingerprint_file.readlines()

    fact_fingerprint_file = open(input_fact)
    fact_fingerprints = fact_fingerprint_file.readlines()

    count_k_fingers = 0
    for i in range(0, len(fingerprints)):

        lengths = fingerprints[i]
        lengths_list = lengths.split()

        facts = fact_fingerprints[i]
        facts_list = facts.split()

        lengths_list = lengths_list[1:]
        facts_list = facts_list[1:]

        for e in range(0, len(lengths_list[:-(k - 1)])):

            count_k_fingers += 1

            k_finger = lengths_list[e:e + k]
            k_finger_fact = facts_list[e:e + k]

            key = ''.join(i + ' ' for i in k_finger)
            key = key[:len(key) - 1]

            key_string = ''.join(i for i in k_finger_fact)

            # Update dict_conflicts
            if key in dict_conflicts:
                key_dict = dict_conflicts[key]

                if key_string in key_dict:
                    count_occurences = key_dict[key_string]
                    count_occurences += 1
                    key_dict[key_string] = count_occurences
                else:
                    key_dict[key_string] = 1
            else:
                key_dict = {key_string: 1}
                dict_conflicts[key] = key_dict

    # Compute conflict
    num_conflicts = 0
    for key in dict_conflicts:
        key_dict = dict_conflicts[key]

        size_dict = len(key_dict)
        '''
        key_conflicts = 1
        for string in key_dict:
            occurences = key_dict[string]
            key_conflicts *= occurences

        num_conflicts = num_conflicts + key_conflicts
        '''
        num_conflicts = num_conflicts + size_dict

    num_keys = len(dict_conflicts)
    print("# k_fingers: ", count_k_fingers)
    print("# keys: ", num_keys)
    print("# Conflicts: ", num_conflicts)

########################################################################################################################
##################################################### MAIN #############################################################
########################################################################################################################
if __name__ == '__main__':

    # Gestione argomenti ###############################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', dest='step', action='store', default='conflicts')
    parser.add_argument('--path', dest='path', action='store', default='training/')
    parser.add_argument('--type_factorization', dest='type_factorization', action='store', default='CFL')
    parser.add_argument('--k_value', dest='k_value', action='store', default=3, type=int)
    parser.add_argument('--file', dest='file', action='store', default="test_majority_result_no_thresholds_list.txt")
    parser.add_argument('--problem', dest='problem', action='store', default="classification")
    parser.add_argument('-n', dest='n', action='store', default=1, type=int)

    args = parser.parse_args()

    # BUILD DATASET
    if args.step == 'conflicts':
        print('\nStatistic Step: COMPUTE # CONFLICTS...\n')
        conflicts_statistic_step(args)

    elif args.step == 'metrics':
        print('\nStatistic Step: COMPUTE METRICS...\n')
        get_metrics(path=args.path, file_results_name=args.file, problem=args.problem)
