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


if __name__ == '__main__':

    # Gestione argomenti ###############################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', dest='file', action='store', default="test_majority_result_no_thresholds_list.txt")
    parser.add_argument('--path', dest='path', action='store', default="testing/")
    parser.add_argument('--problem', dest='problem', action='store', default="classification")


    args = parser.parse_args()

    print('\nCompute metrics...\n')
    get_metrics(path=args.path, file_results_name=args.file,problem=args.problem)

