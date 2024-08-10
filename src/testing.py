import argparse
import logging
import os
import pickle
import operator
import statistics

from functools import partial
from multiprocessing.pool import Pool

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from fingerprint_utils import extract_reads_mp, compute_split_fingerprint_by_list
from machine_learning_utils import test_reads_fusion
from factorizations import CFL, ICFL_recursive, CFL_icfl
from factorizations_comb import d_cfl, d_icfl, d_cfl_icfl
import csv

global dataset_name_fastq


class MetricsCounter:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        self.accuracy = 0
        self.recall = 0
        self.precision = 0
        self.specificity = 0
        self.f1_score = 0

        self.chimeric = 0
        self.non_chimeric = 0

    # Called in positive cases
    def increment_truePositive(self, fusion_gene_1, fusion_gene_2):

        # If 'chimeric' (2 different genes) and there is a fusion, is 'True Positive'
        if fusion_gene_1 != fusion_gene_2:
            self.tp += 1
            self.chimeric += 1

    def increment_falsePositive(self, fusion_gene_1, fusion_gene_2):

        # If 'not chimeric' (2 identical genes) and there is a fusion, is 'False positive'
        if fusion_gene_1 == fusion_gene_2:
            self.fp += 1
            # self.non_chimeric += 1

    def increment_trueNegative(self, fusion_gene_1, fusion_gene_2):

        # if is 'not chimeric' (2 identical genes) and there is not a fusion, is 'True Negative'
        if fusion_gene_1 == fusion_gene_2:
            self.tn += 1
            self.non_chimeric += 1

    def increment_falseNegative(self, fusion_gene_1, fusion_gene_2):

        # if 'chimeric' (2 different genes) and there is not a fusion, is 'False negative'
        if fusion_gene_1 != fusion_gene_2:
            self.fn += 1
            # self.chimeric += 1

    def print_num_chimeric_nonChimeric(self):
        print("CHIMERIC: ", self.chimeric, "\nNON CHIMERIC: ", self.non_chimeric)

    def print_raw_metrics(self):

        print("RAW METRICS: TP: [", self.tp, "] FP: [", self.fp, "] TN: [", self.tn, "] FN: [", self.fn, ']')

    # Calculate metrics for dataset chimeric/nonChimeric
    def calculate_metrics(self):
        # Calculate accuracy
        self.accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn) if (
                                                                                                 self.tp + self.fp + self.tn + self.fn) != 0 else 0.0

        # Calculate recall
        self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) != 0 else 0.0

        # Calculate precision
        self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) != 0 else 0.0

        # Calculate F1-score
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall) if (
                                                                                                       self.precision + self.recall) != 0 else 0.0

        # Calculate specificity
        self.specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) != 0 else 0.0

        # Le tue informazioni
        metrics_string = (
            f"Accuracy: {self.accuracy:.2f} | "
            f"Recall: {self.recall:.2f} | "
            f"Precision: {self.precision:.2f} | "
            f"F1-score: {self.f1_score:.2f} | "
            f"Specificity: {self.specificity:.2f}"
        )

        return metrics_string

    def save_csv_metric(self, output_csv_path=None):
        # Save metrics to CSV file if not not None
        if output_csv_path:
            with open(output_csv_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Metric', 'Value'])
                csv_writer.writerow(['Accuracy', f'{self.accuracy:.2f}'])
                csv_writer.writerow(['Recall', f'{self.recall:.2f}'])
                csv_writer.writerow(['Precision', f'{self.precision:.2f}'])
                csv_writer.writerow(['F1-score', f'{self.f1_score:.2f}'])
                csv_writer.writerow(['Specificity', f'{self.specificity:.2f}'])


# Given a set of reads, performs classification by using the majority (or thresholds) criterion on best k-finger classification
# args.step = 'test_fusion' ##########################################################################################
def testing_reads_fusion_mp_step(args, dataset_path, dataset_name_fasta):
    input_fasta = dataset_path + dataset_name_fasta
    increment = 2  # 4 if: @43eccc5e-69fb-403d-a19a-99c10b87f829..\nATGAAGATT..\n+\n/&*#'*//,///...
    print(args.n)
    # Apre il file FASTA specificato in input_fasta e legge le linee del file
    print(input_fasta)
    file = open(input_fasta)
    lines = file.readlines()

    # Le linee lette dal file sono divise in parti, dove ogni parte contiene 'increment' linee.
    splitted_read_lines = [lines[i:i + increment] for i in range(0, len(lines), increment)]

    # Le parti sono suddivise ulteriormente in base al numero di processi paralleli
    splitted_read_lines_for_process = [splitted_read_lines[i:i + int(len(splitted_read_lines) / args.n)] for i in
                                       range(0, len(splitted_read_lines), int(len(splitted_read_lines) / args.n))]

    # Se ci sono più parti di quelle necessarie per il numero
    # di processi paralleli, l'ultima parte viene aggiunta alla penultima.
    if len(splitted_read_lines_for_process) > 1:
        obj_for_proc = len(splitted_read_lines) / args.n
        if isinstance(obj_for_proc, float):
            splitted_read_lines_for_process[len(splitted_read_lines_for_process) - 2] = splitted_read_lines_for_process[
                                                                                            len(
                                                                                                splitted_read_lines_for_process) - 2] + \
                                                                                        splitted_read_lines_for_process[
                                                                                            len(
                                                                                                splitted_read_lines_for_process) - 1]

            splitted_read_lines_for_process = splitted_read_lines_for_process[:len(splitted_read_lines_for_process) - 1]

    # Il codice crea un pool di processi paralleli (Pool) e mappa la funzione
    # schema_testing_reads_fusion_mp sui dati divisi. I risultati sono poi aggregati.
    with Pool(args.n) as pool:
        func = partial(schema_testing_reads_fusion_mp, args, dataset_path, dataset_name_fasta)

        read_lines = []
        for res in pool.map(func, splitted_read_lines_for_process):
            read_lines = read_lines + res

    # Le linee risultanti sono modificate (trasformate in maiuscolo) e scritte
    # in un nuovo file di testo (test_fusion_result_CFL_ICFL_COMB-30_K8.txt).
    read_lines = [s.upper() for s in read_lines]

    # Results txt file
    dataset_name = dataset_name_fasta.replace(".fastq", "")
    test_result_file = open(dataset_path + "test_fusion_result_CFL_ICFL_COMB-30_K8_" + dataset_name + ".txt", 'w')
    test_result_file.writelines(read_lines)
    test_result_file.close()


########################################################################################################################

def schema_testing_reads_fusion_mp(args, dataset_path, dataset_name_fasta, reads=[]):
    input_fasta = dataset_path + dataset_name_fasta

    # EXTRACT READS ####################################################################################################
    read_lines = extract_reads_mp(input_fasta, reads)

    if len(read_lines) == 0:
        print('No reads extracted!')
        read_lines = []
        return read_lines

    read_lines = [s.upper() for s in read_lines]
    # print(read_lines)

    print('# READS: ', len(read_lines))

    # COMPUTE FINGERPRINTS #############################################################################################
    type_factorization = args.type_factorization

    # Check type factorization
    factorization = None
    T = None
    if type_factorization == "CFL":
        factorization = CFL
    elif type_factorization == "ICFL":
        factorization = ICFL_recursive
    elif type_factorization == "CFL_ICFL-10":
        factorization = CFL_icfl
        T = 10
    elif type_factorization == "CFL_ICFL-20":
        factorization = CFL_icfl
        T = 20
    elif type_factorization == "CFL_ICFL-30":
        factorization = CFL_icfl
        T = 30
    elif type_factorization == "CFL_COMB":
        factorization = d_cfl
    elif type_factorization == "ICFL_COMB":
        factorization = d_icfl
    elif type_factorization == "CFL_ICFL_COMB-10":
        factorization = d_cfl_icfl
        T = 10
    elif type_factorization == "CFL_ICFL_COMB-20":
        factorization = d_cfl_icfl
        T = 20
    elif type_factorization == "CFL_ICFL_COMB-30":
        factorization = d_cfl_icfl
        T = 30

    dictionary = None
    dictionary_lines = None
    if args.dictionary == 'yes':
        dictionary = open("%s" % args.path + "dictionary_" + args.type_factorization + ".txt", 'r')
        dictionary_lines = dictionary.readlines()
        dictionary_lines = [line.replace('\n', '') for line in dictionary_lines]

    res = compute_split_fingerprint_by_list(args.fact, factorization, T, args.dictionary, dictionary_lines, read_lines)
    ####################################################################################################################

    # TEST READS #######################################################################################################
    # Best model
    best_model_path = args.path + args.best_model
    list_best_model = pickle.load(open(best_model_path, "rb"))

    res = test_reads_fusion(list_best_model, args.path, args.type_factorization, args.k_value, res)
    ####################################################################################################################

    return res


# args.step = 'analyze_test_fusion' ##########################################################################################
# def analyze_reads_fusion_mp_step(args):
def gene_fusion_count(path, result_file, type_factorization, dataset_name_fastq):
    # result_file_path = args.path + args.result_file

    dataset_name = dataset_name_fastq.replace(".fastq", "")

    result_file_path = path + result_file
    result_file = open(result_file_path)

    result_lines = result_file.readlines()

    # Remove '['
    result_lines = [s.replace('[', '') for s in result_lines]

    # Remove ']'
    result_lines = [s.replace(']', '') for s in result_lines]

    # Remove '\n'
    result_lines = [s.replace('\n', '') for s in result_lines]

    # file = open("%s" % args.path + "gene_fusion_count_" + args.type_factorization + ".txt", 'w')
    gene_fusion_count_file = open("%s" % path + "gene_fusion_count_" + type_factorization + "_" + dataset_name + ".txt",
                                  'w')
    gene_fusion_count_lines = []

    threshold = 95
    count_fusion = 0

    for line in result_lines:
        line = line.split('- PREDICTION:')
        line = line[1]

        line = line.split()

        line_dictionary = {}
        for s in line:
            if s in line_dictionary:
                count = line_dictionary[s]
                count = count + 1
                line_dictionary[s] = count
            else:
                line_dictionary[s] = 1

        # sort dictionary by value
        line_dictionary = {k: v for k, v in sorted(line_dictionary.items(), key=lambda item: item[1])}

        list_genes_counted = []
        for key, value in line_dictionary.items():
            s = str(key) + ':' + str(value)
            list_genes_counted.append(s)

        list_genes_counted = list_genes_counted[::-1]

        new_line = ' '.join(str(gene_count) + ' - ' for gene_count in list_genes_counted) + '\n'
        gene_fusion_count_lines.append(new_line)

    gene_fusion_count_file.writelines(gene_fusion_count_lines)
    gene_fusion_count_file.close()


def parse_gene_fusion_result(testing_path, dataset_name_fastq):
    dataset_name = dataset_name_fastq.replace(".fastq", "")

    # Creazione dizionario
    file_genes = open('testing/RF_kfinger_clsf_report_CFL_ICFL_COMB-30_K8.csv')

    genes_lines = file_genes.readlines()
    genes_dictionary = {}
    for i in range(1, len(genes_lines)):
        line = genes_lines[i]
        line = line.split(',')

        value = line[0]
        key = i - 1

        genes_dictionary[key] = value

    file_genes.close()

    # parse results
    file_result = open(testing_path + 'gene_fusion_count_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt')
    result_lines = file_result.readlines()

    file_parsed = open(testing_path + 'parsed_gene_fusion_count_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt', 'w')
    parsed_lines = []

    for line in result_lines:
        line = line.split('-')

        new_line = ''
        for l in line:
            l = l.split(':')
            if l[0] == ' \n':
                continue
            id_gene = int(l[0])
            count_gene = l[1]

            label_gene = genes_dictionary[id_gene]
            new_line = new_line + label_gene + ':' + str(count_gene) + ' - '

        parsed_lines.append(new_line[:len(new_line) - 3] + '\n')

    file_parsed.writelines(parsed_lines)
    file_parsed.close()


def analyze_gene_fusion(testing_path, dataset_name_fastq):
    dataset_name = dataset_name_fastq.replace(".fastq", "")

    # I 2 files hanno una corrispondenza biunivoca: riga i in "file_parsed" corrisponde a riga i in "file_result"
    file_parsed = open(testing_path + 'parsed_gene_fusion_count_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt')
    file_result = open(testing_path + 'test_fusion_result_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt')
    file_no_criterion = open(testing_path + 'no_fusion_criterion_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt', 'w')

    parsed_lines = file_parsed.readlines()
    result_lines = file_result.readlines()
    no_criterion_lines = []

    count_checked = 0
    i = 1
    for parsed_line, result_line in zip(parsed_lines, result_lines):

        # Estraiamo i 2 geni di fusione da "file_result" ###############################################################
        l = result_line.split()
        l = l[1].split('--')

        l1 = l[0].split('|')
        l1 = l1[1].split('.')

        fusion_gene_1 = l1[0]

        l2 = l[1].split('|')
        l2 = l2[1].split('.')

        fusion_gene_2 = l2[0]
        ################################################################################################################

        # 2) Controlliamo se i 2 geni sono tra i primi 3 elementi in "parsed_file" #####################################
        parsed = parsed_line.split('-')

        l1 = parsed[0].split(':')
        parsed_gene_1 = l1[0]
        parsed_gene_1 = parsed_gene_1.split()[0]

        if len(parsed) == 1:
            no_criterion_lines.append(result_line)
            no_criterion_lines.append(parsed_line)
            continue

        # print(parsed)

        l2 = parsed[1].split(':')

        parsed_gene_2 = l2[0]
        parsed_gene_2 = parsed_gene_2.split()[0]

        if len(parsed) > 2:
            l3 = parsed[2].split(':')
            parsed_gene_3 = l3[0]
            parsed_gene_3 = parsed_gene_3.split()[0]

        check_list = []
        if len(parsed) > 2:
            check_list = [parsed_gene_1, parsed_gene_2, parsed_gene_3]
        else:
            check_list = [parsed_gene_1, parsed_gene_2]

        # controlliamo se fusion_gene_1 e fusion_gene_2 sono in check_list
        if fusion_gene_1 in check_list and fusion_gene_2 in check_list:
            count_checked += 1
        else:
            no_criterion_lines.append('i: ' + str(i) + '\n')
            no_criterion_lines.append(result_line)
            no_criterion_lines.append(parsed_line + '\n')

        i = i + 1
        ################################################################################################################

    accuracy = float((count_checked * 100) / len(parsed_lines))
    print('% of fusion genes into the first 3 positions: {}'.format(accuracy))

    file_no_criterion.writelines(no_criterion_lines)

    file_parsed.close()
    file_result.close()
    file_no_criterion.close()


def statistical_analysis_with_break_index(testing_path, num_line_for_read, dataset_name_fastq="reads-both.fastq"):
    dataset_name = dataset_name_fastq.replace(".fastq", "")
    # Pre-processing ###################################################################################################
    # open files
    file_test = open(testing_path + 'test_fusion_result_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt')
    file_parsed = open(testing_path + 'parsed_gene_fusion_count_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt')
    file_reads = open(testing_path + dataset_name_fastq)

    # create new file
    file_statistics = open(testing_path + 'statistics_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt', 'w')

    # read lines
    test_lines = file_test.readlines()
    parsed_lines = file_parsed.readlines()
    reads_lines = file_reads.readlines()

    statistics_lines = []

    # Creazione dizionario
    file_genes = open('testing/RF_kfinger_clsf_report_CFL_ICFL_COMB-30_K8.csv')

    genes_lines = file_genes.readlines()
    genes_dictionary = {}
    for i in range(1, len(genes_lines)):
        line = genes_lines[i]
        line = line.split(',')

        value = line[0]
        key = i - 1

        genes_dictionary[key] = value

    file_genes.close()
    ####################################################################################################################

    # For each read create the lines in 'statistics_CFL_ICFL_COMB-30_K8.txt'
    for i in range(len(test_lines)):

        test_line = test_lines[i]
        parsed_line = parsed_lines[i]

        # Find corresponding read in 'dataset_name_fastq' ################################################################
        start_index = i * num_line_for_read
        end_index = start_index + num_line_for_read
        lines_for_current_read = reads_lines[start_index:end_index]
        read_line = lines_for_current_read[1]

        # INDIVIDUAZIONE break fusion index -- "DA CAMBIARE CON IL DATO DI YURI" #######################################
        break_fusion_index = -1  # posizione nella read in cui comincia finisce un gene ed inizia l'altro
        for j, c in enumerate(read_line):
            if not c.isupper():
                break_fusion_index = j
                break
        ################################################################################################################

        # Find most occurrent Gene_1 BEFORE break_fusion_index and Gene_2 AFTER break_fusion_index #####################
        # a) get list of kfinger predictions from test_line
        test_line_lst = test_line.split(' - PREDICTION: ')

        # b) splittiamo tra la prima parte contenente la sequenza di lunghezze e la seconda parte contenente le predictions
        # l'i-esimo elemento di kfingers_line_lst corrisponde all'i-esimo elemento di gene_line_lst
        kfingers_line_lst = test_line_lst[0]
        gene_line_lst = test_line_lst[1]

        kfingers_line_lst = kfingers_line_lst.split('|')
        kfingers_line_lst = kfingers_line_lst[3:]

        gene_line_lst = gene_line_lst.split('] [')
        gene_line_lst = gene_line_lst[1:]

        # splittiamo i geni contenuti in test_line_lst in 2 liste: prima di break_fusion_index e dopo break_fusion_index
        genes_before_break_fusion = []
        genes_after_break_fusion = []

        current_position = 0

        for kfinger_line, gene_line in zip(kfingers_line_lst, gene_line_lst):
            kfinger_line = kfinger_line.split()
            gene_line = gene_line.split()
            for j, g in enumerate(gene_line):
                if j < len(gene_line) - 1:
                    current_position += int(kfinger_line[j])
                else:
                    for k in range(j, len(kfinger_line)):
                        current_position += int(kfinger_line[k])

                current_g = g.replace('[', '')
                current_g = current_g.replace(']', '')
                current_g = current_g.replace(' ', '')

                label_gene = genes_dictionary[int(current_g)]

                if current_position <= break_fusion_index:
                    genes_before_break_fusion.append(label_gene)
                else:
                    genes_after_break_fusion.append(label_gene)

        # c) count genes in genes_before_break_fusion and genes_after_break_fusion

        gene_before = most_common(genes_before_break_fusion)
        gene_after = most_common(genes_after_break_fusion)
        ################################################################################################################

        ################################################################################################################
        # Add lines in 'statistics_CFL_ICFL_COMB-30_K8.txt'
        read_line = read_line.replace('\n', '')
        test_line = test_line.replace('\n', '')
        parsed_line = parsed_line.replace('\n', '')

        statistics_lines.append('READ: ' + read_line + '\n')
        statistics_lines.append('TEST_RESULT: ' + test_line + '\n')
        statistics_lines.append('SORTING GENES: ' + parsed_line + '\n')

        stastistics_line = 'STATISTICS: break_fusion_position: ' + str(break_fusion_index) + ' $ '

        # Controlla che la lista dei geni piu' frequenti PRIMA del punto di break non sia nulla
        if len(gene_before) != 0:
            stastistics_line = stastistics_line + 'first_fusion_gene: ' + gene_before + ' $ '

        # Controlla che la lista dei geni piu' frequenti DOPO il punto di break non sia nulla
        if len(gene_after) != 0:
            stastistics_line = stastistics_line + 'second_fusion_gene: ' + gene_after + '\n\n'

        statistics_lines.append(stastistics_line)

    file_statistics.writelines(statistics_lines)

    # Close files ######################################################################################################
    file_test.close()
    file_parsed.close()
    file_reads.close()
    file_statistics.close()


########################################################################################################################

def most_common(lst):
    # CONTROLLA CHE LA LISTA DI GENI IN INPUT(PRIMA DEL BREAK O DOPO IL BREAK) NON SIA NULL
    if len(lst) != 0:
        return max(set(lst), key=lst.count)
    else:
        return []


def most_consecutive_frequent(lst):
    # Create genes dictionary and initialize each to 0
    g_dictionary = {}
    for g in lst:
        value = 0
        key = g

        g_dictionary[key] = value

    # Count consecutive frequency for each gene
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
            value = g_dictionary[lst[i]]
            value += 1
            g_dictionary[lst[i]] = value

    return max(g_dictionary.items(), key=operator.itemgetter(1))[0]


def statistical_analysis_with_known_genes_check_range_majority(testing_path, statistics_path, num_line_for_read,
                                                               dataset_name_fastq="reads-both.fastq",
                                                               fusion_threshold=60, metrics_counter=MetricsCounter()):
    # Pre-processing ###################################################################################################
    # open files
    dataset_name = dataset_name_fastq.replace(".fastq", "")

    file_test = open(testing_path + 'test_fusion_result_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt')
    file_parsed = open(testing_path + 'parsed_gene_fusion_count_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt')
    file_reads = open(testing_path + dataset_name_fastq)

    # create new file
    file_statistics = open(statistics_path, 'w')

    # read lines
    test_lines = file_test.readlines()
    parsed_lines = file_parsed.readlines()
    reads_lines = file_reads.readlines()

    statistics_lines = []

    # Creazione dizionario
    file_genes = open('testing/RF_kfinger_clsf_report_CFL_ICFL_COMB-30_K8.csv')

    genes_lines = file_genes.readlines()
    genes_dictionary = {}
    for i in range(1, len(genes_lines)):
        line = genes_lines[i]
        line = line.split(',')

        value = line[0]
        key = i - 1

        genes_dictionary[key] = value

    file_genes.close()
    ####################################################################################################################

    # For each read create the lines in 'statistics_CFL_ICFL_COMB-30_K8.txt'
    for i in range(len(test_lines)):

        test_line = test_lines[i]
        parsed_line = parsed_lines[i]

        # Find corresponding read in 'dataset_name_fastq' ################################################################
        start_index = i * num_line_for_read
        end_index = start_index + num_line_for_read
        lines_for_current_read = reads_lines[start_index:end_index]
        read_line = lines_for_current_read[1]

        # Find most occurrent Gene_1 BEFORE break_fusion_index and Gene_2 AFTER break_fusion_index #####################
        # a) get list of kfinger predictions from test_line
        test_line_lst = test_line.split(' - PREDICTION: ')

        # b) splittiamo tra la prima parte contenente la sequenza di lunghezze e la seconda parte contenente le predictions
        kfingers_line_lst = test_line_lst[0]
        gene_line_lst = test_line_lst[1]

        # Extract the 2 reference fusion genes #########################################################################
        kfingers_line_lst = kfingers_line_lst.split('|')

        # Fusion Gene 1
        fusion_gene_1 = kfingers_line_lst[1]
        fusion_gene_1 = fusion_gene_1.split('.')
        fusion_gene_1 = fusion_gene_1[0]

        # Fusion Gene 2
        fusion_gene_2 = kfingers_line_lst[2]
        fusion_gene_2 = fusion_gene_2.split('.')
        fusion_gene_2 = fusion_gene_2[0]
        ################################################################################################################

        gene_line_lst = gene_line_lst.replace('[', '')
        gene_line_lst = gene_line_lst.replace(']', '')
        gene_line_lst = gene_line_lst.split()
        gene_line_lst = [genes_dictionary[int(g)] for g in gene_line_lst]

        # Add lines in 'statistics_CFL_ICFL_COMB-30_K8.txt'
        read_line = read_line.replace('\n', '')
        test_line = test_line.replace('\n', '')
        parsed_line = parsed_line.replace('\n', '')

        statistics_lines.append('READ: ' + read_line + '\n')
        statistics_lines.append('TEST_RESULT: ' + test_line + '\n')
        statistics_lines.append('SORTING GENES: ' + parsed_line + '\n')

        # Check relation between "gene 1 interval" and "gene 2 interval" ###############################################
        try:
            start_index_gene_1 = gene_line_lst.index(fusion_gene_1)
            end_index_gene_1 = len(gene_line_lst) - gene_line_lst[::-1].index(fusion_gene_1) - 1

            start_index_gene_2 = gene_line_lst.index(fusion_gene_2)
            end_index_gene_2 = len(gene_line_lst) - gene_line_lst[::-1].index(fusion_gene_2) - 1

        except ValueError:

            # stastistics_line = 'NO Break Fusion Interval!' + '\n'
            # stastistics_line = stastistics_line + '-\n'
            # stastistics_line = stastistics_line + '-\n\n'
            # statistics_lines.append(stastistics_line)

            # TN CASE
            # metrics_counter.increment_trueNegative(fusion_gene_1, fusion_gene_2)

            continue

        range_gene_1 = []
        range_gene_2 = []

        # FIRST CASE
        if start_index_gene_1 < end_index_gene_1 and start_index_gene_2 < end_index_gene_2 and end_index_gene_1 < end_index_gene_2:

            end_1 = 0
            if end_index_gene_1 == len(gene_line_lst):
                end_1 = end_index_gene_1
            else:
                end_1 = end_index_gene_1 + 1
            range_gene_1 = gene_line_lst[start_index_gene_1:end_1]
            range_gene_1 = smooth_range(lst=range_gene_1, threshold=1)

            end_2 = 0
            if end_index_gene_2 == len(gene_line_lst):
                end_2 = end_index_gene_2
            else:
                end_2 = end_index_gene_2 + 1
            range_gene_2 = gene_line_lst[start_index_gene_2:end_2]
            range_gene_2 = smooth_range(lst=range_gene_2, threshold=1)

            most_common_gene_in_range_gene_1 = most_common(range_gene_1)
            most_common_gene_in_range_gene_2 = most_common(range_gene_2)

            # Compute fusion_score
            perc_1 = ((end_1 - start_index_gene_1 + 1) * 100) / len(gene_line_lst)
            perc_2 = ((end_2 - start_index_gene_2 + 1) * 100) / len(gene_line_lst)
            perc_intersec = (len(gene_line_lst[start_index_gene_2:end_1]) * 100) / len(gene_line_lst)
            fusion_score = abs(100 - (perc_1 + perc_2 - perc_intersec))

            if fusion_score >= fusion_threshold:  # IL MODELLO DICE CHE E' CHIMERICO

                if fusion_gene_1 == fusion_gene_2:

                    stastistics_line = 'NO Break Fusion!- fusion_score:' + str(fusion_score) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                        start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_2:' + str(
                        fusion_gene_2) + ' |start_index:' + str(
                        start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                    # stastistics_line = stastistics_line + '-\n'
                    # stastistics_line = stastistics_line + '-\n\n'
                    statistics_lines.append(stastistics_line)

                    metrics_counter.increment_falsePositive(fusion_gene_1, fusion_gene_2)


                elif most_common_gene_in_range_gene_1 == fusion_gene_1 and most_common_gene_in_range_gene_2 == fusion_gene_2:

                    stastistics_line = 'YES Break Fusion - fusion_score:' + str(fusion_score) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                        start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_2:' + str(
                        fusion_gene_2) + ' |start_index:' + str(
                        start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                    statistics_lines.append(stastistics_line)

                    metrics_counter.increment_truePositive(fusion_gene_1, fusion_gene_2)

            elif fusion_score < fusion_threshold:

                if fusion_gene_1 == fusion_gene_2:

                    stastistics_line = 'NO Break Fusion! - fusion_score:' + str(fusion_score) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                        start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_2:' + str(
                        fusion_gene_2) + ' |start_index:' + str(
                        start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                    # stastistics_line = stastistics_line + '-\n'
                    # stastistics_line = stastistics_line + '-\n\n'
                    statistics_lines.append(stastistics_line)

                    metrics_counter.increment_trueNegative(fusion_gene_1, fusion_gene_2)


                elif most_common_gene_in_range_gene_1 == fusion_gene_1 and most_common_gene_in_range_gene_2 == fusion_gene_2:

                    stastistics_line = 'YES Break Fusion - fusion_score:' + str(fusion_score) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                        start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_2:' + str(
                        fusion_gene_2) + ' |start_index:' + str(
                        start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                    statistics_lines.append(stastistics_line)

                    metrics_counter.increment_falseNegative(fusion_gene_1, fusion_gene_2)

        # SECOND CASE
        elif start_index_gene_2 < end_index_gene_2 and start_index_gene_1 < end_index_gene_1 and end_index_gene_2 < end_index_gene_1:

            end_1 = 0
            if end_index_gene_1 == len(gene_line_lst):
                end_1 = end_index_gene_1
            else:
                end_1 = end_index_gene_1 + 1
            range_gene_1 = gene_line_lst[start_index_gene_1:end_1]

            end_2 = 0
            if end_index_gene_2 == len(gene_line_lst):
                end_2 = end_index_gene_2
            else:
                end_2 = end_index_gene_2 + 1
            range_gene_2 = gene_line_lst[start_index_gene_2:end_2]

            most_common_gene_in_range_gene_1 = most_common(range_gene_1)
            most_common_gene_in_range_gene_2 = most_common(range_gene_2)

            # Compute fusion_score
            perc_1 = ((end_1 - start_index_gene_1 + 1) * 100) / len(gene_line_lst)
            perc_2 = ((end_2 - start_index_gene_2 + 1) * 100) / len(gene_line_lst)
            perc_intersec = (len(gene_line_lst[start_index_gene_2:end_1]) * 100) / len(gene_line_lst)
            fusion_score = abs(100 - (perc_1 + perc_2 - perc_intersec))

            if fusion_score >= fusion_threshold:  # IL MODELLO DICE CHE E' CHIMERICO

                if fusion_gene_1 == fusion_gene_2:

                    stastistics_line = 'NO Break Fusion!- fusion_score:' + str(fusion_score) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                        start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_2:' + str(
                        fusion_gene_2) + ' |start_index:' + str(
                        start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                    # stastistics_line = stastistics_line + '-\n'
                    # stastistics_line = stastistics_line + '-\n\n'
                    statistics_lines.append(stastistics_line)

                    metrics_counter.increment_falsePositive(fusion_gene_1, fusion_gene_2)


                elif most_common_gene_in_range_gene_1 == fusion_gene_1 and most_common_gene_in_range_gene_2 == fusion_gene_2:

                    stastistics_line = 'YES Break Fusion - fusion_score:' + str(fusion_score) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                        start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_2:' + str(
                        fusion_gene_2) + ' |start_index:' + str(
                        start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                    statistics_lines.append(stastistics_line)

                    metrics_counter.increment_truePositive(fusion_gene_1, fusion_gene_2)

            elif fusion_score < fusion_threshold:

                if fusion_gene_1 == fusion_gene_2:

                    stastistics_line = 'NO Break Fusion! - fusion_score:' + str(fusion_score) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                        start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_2:' + str(
                        fusion_gene_2) + ' |start_index:' + str(
                        start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                    # stastistics_line = stastistics_line + '-\n'
                    # stastistics_line = stastistics_line + '-\n\n'
                    statistics_lines.append(stastistics_line)

                    metrics_counter.increment_trueNegative(fusion_gene_1, fusion_gene_2)


                elif most_common_gene_in_range_gene_1 == fusion_gene_1 and most_common_gene_in_range_gene_2 == fusion_gene_2:

                    stastistics_line = 'YES Break Fusion - fusion_score:' + str(fusion_score) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                        start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                    stastistics_line = stastistics_line + 'fusion_gene_2:' + str(
                        fusion_gene_2) + ' |start_index:' + str(
                        start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                    statistics_lines.append(stastistics_line)

                    metrics_counter.increment_falseNegative(fusion_gene_1, fusion_gene_2)

    ################################################################################################################

    ################################################################################################################

    file_statistics.writelines(statistics_lines)

    # Close files ######################################################################################################
    file_test.close()
    file_parsed.close()
    file_reads.close()
    file_statistics.close()

    return metrics_counter


########################################################################################################################


def statistical_analysis_with_known_genes_no_check_range_majority(testing_path, statistics_path, num_line_for_read,
                                                                  dataset_name_fastq="reads-both.fastq",
                                                                  fusion_threshold=60,
                                                                  metrics_counter=MetricsCounter()):
    # Pre-processing ###################################################################################################
    # open files
    dataset_name = dataset_name_fastq.replace(".fastq", "")

    file_test = open(testing_path + 'test_fusion_result_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt')
    file_parsed = open(testing_path + 'parsed_gene_fusion_count_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt')
    file_reads = open(testing_path + dataset_name_fastq)

    # create new file
    file_statistics = open(statistics_path, 'w')

    # read lines
    test_lines = file_test.readlines()
    parsed_lines = file_parsed.readlines()
    reads_lines = file_reads.readlines()

    statistics_lines = []

    # Creazione dizionario
    file_genes = open('testing/RF_kfinger_clsf_report_CFL_ICFL_COMB-30_K8.csv')

    genes_lines = file_genes.readlines()
    genes_dictionary = {}
    for i in range(1, len(genes_lines)):
        line = genes_lines[i]
        line = line.split(',')

        value = line[0]
        key = i - 1

        genes_dictionary[key] = value

    file_genes.close()
    ####################################################################################################################

    # For each read create the lines in 'statistics_CFL_ICFL_COMB-30_K8.txt'
    for i in range(len(test_lines)):

        test_line = test_lines[i]
        parsed_line = parsed_lines[i]

        # Find corresponding read in 'reads-both.fastq' ################################################################
        start_index = i * num_line_for_read
        end_index = start_index + num_line_for_read
        lines_for_current_read = reads_lines[start_index:end_index]
        read_line = lines_for_current_read[1]

        # Find most occurrent Gene_1 BEFORE break_fusion_index and Gene_2 AFTER break_fusion_index #####################
        # a) get list of kfinger predictions from test_line
        test_line_lst = test_line.split(' - PREDICTION: ')

        # b) splittiamo tra la prima parte contenente la sequenza di lunghezze e la seconda parte contenente le predictions
        kfingers_line_lst = test_line_lst[0]
        gene_line_lst = test_line_lst[1]

        # Extract the 2 reference fusion genes #########################################################################
        kfingers_line_lst = kfingers_line_lst.split('|')

        # Fusion Gene 1
        fusion_gene_1 = kfingers_line_lst[1]
        fusion_gene_1 = fusion_gene_1.split('.')
        fusion_gene_1 = fusion_gene_1[0]

        # Fusion Gene 2
        fusion_gene_2 = kfingers_line_lst[2]
        fusion_gene_2 = fusion_gene_2.split('.')
        fusion_gene_2 = fusion_gene_2[0]
        ################################################################################################################

        gene_line_lst = gene_line_lst.replace('[', '')
        gene_line_lst = gene_line_lst.replace(']', '')
        gene_line_lst = gene_line_lst.split()
        gene_line_lst = [genes_dictionary[int(g)] for g in gene_line_lst]

        # Add lines in 'statistics_CFL_ICFL_COMB-30_K8.txt'
        read_line = read_line.replace('\n', '')
        test_line = test_line.replace('\n', '')
        parsed_line = parsed_line.replace('\n', '')

        statistics_lines.append('READ: ' + read_line + '\n')
        statistics_lines.append('TEST_RESULT: ' + test_line + '\n')
        statistics_lines.append('SORTING GENES: ' + parsed_line + '\n')

        # Check relation between "gene 1 interval" and "gene 2 interval" ###############################################
        try:
            start_index_gene_1 = gene_line_lst.index(fusion_gene_1)
            end_index_gene_1 = len(gene_line_lst) - gene_line_lst[::-1].index(fusion_gene_1) - 1

            start_index_gene_2 = gene_line_lst.index(fusion_gene_2)
            end_index_gene_2 = len(gene_line_lst) - gene_line_lst[::-1].index(fusion_gene_2) - 1

        except ValueError:

            # stastistics_line = 'NO Break Fusion Interval!' + '\n'
            # stastistics_line = stastistics_line + '-\n'
            # stastistics_line = stastistics_line + '-\n\n'
            # statistics_lines.append(stastistics_line)

            # TN CASE
            # metrics_counter.increment_trueNegative(fusion_gene_1, fusion_gene_2)

            continue

        range_gene_1 = []
        range_gene_2 = []

        end_1 = 0
        if end_index_gene_1 == len(gene_line_lst):
            end_1 = end_index_gene_1
        else:
            end_1 = end_index_gene_1 + 1
        range_gene_1 = gene_line_lst[start_index_gene_1:end_1]
        range_gene_1 = smooth_range(lst=range_gene_1, threshold=1)

        end_2 = 0
        if end_index_gene_2 == len(gene_line_lst):
            end_2 = end_index_gene_2
        else:
            end_2 = end_index_gene_2 + 1

        range_gene_2 = gene_line_lst[start_index_gene_2:end_2]
        range_gene_2 = smooth_range(lst=range_gene_2, threshold=1)

        most_common_gene_in_range_gene_1 = most_common(range_gene_1)
        most_common_gene_in_range_gene_2 = most_common(range_gene_2)

        # Compute fusion_score
        perc_1 = ((end_1 - start_index_gene_1 + 1) * 100) / len(gene_line_lst)
        perc_2 = ((end_2 - start_index_gene_2 + 1) * 100) / len(gene_line_lst)
        perc_intersec = (len(gene_line_lst[start_index_gene_2:end_1]) * 100) / len(gene_line_lst)
        fusion_score = abs(100 - (perc_1 + perc_2 - perc_intersec))

        if fusion_score >= fusion_threshold:  # IL MODELLO DICE CHE E' CHIMERICO

            if fusion_gene_1 == fusion_gene_2:

                stastistics_line = 'NO Break Fusion!- fusion_score:' + str(fusion_score) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                    start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_2:' + str(fusion_gene_2) + ' |start_index:' + str(
                    start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                # stastistics_line = stastistics_line + '-\n'
                # stastistics_line = stastistics_line + '-\n\n'
                statistics_lines.append(stastistics_line)

                metrics_counter.increment_falsePositive(fusion_gene_1, fusion_gene_2)


            elif most_common_gene_in_range_gene_1 == fusion_gene_1 and most_common_gene_in_range_gene_2 == fusion_gene_2:

                stastistics_line = 'YES Break Fusion - fusion_score:' + str(fusion_score) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                    start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_2:' + str(fusion_gene_2) + ' |start_index:' + str(
                    start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                statistics_lines.append(stastistics_line)

                metrics_counter.increment_truePositive(fusion_gene_1, fusion_gene_2)

        elif fusion_score < fusion_threshold:

            if fusion_gene_1 == fusion_gene_2:

                stastistics_line = 'NO Break Fusion! - fusion_score:' + str(fusion_score) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                    start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_2:' + str(fusion_gene_2) + ' |start_index:' + str(
                    start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                # stastistics_line = stastistics_line + '-\n'
                # stastistics_line = stastistics_line + '-\n\n'
                statistics_lines.append(stastistics_line)

                metrics_counter.increment_trueNegative(fusion_gene_1, fusion_gene_2)


            elif most_common_gene_in_range_gene_1 == fusion_gene_1 and most_common_gene_in_range_gene_2 == fusion_gene_2:

                stastistics_line = 'YES Break Fusion - fusion_score:' + str(fusion_score) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                    start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_2:' + str(fusion_gene_2) + ' |start_index:' + str(
                    start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                statistics_lines.append(stastistics_line)

                metrics_counter.increment_falseNegative(fusion_gene_1, fusion_gene_2)

        ################################################################################################################

    file_statistics.writelines(statistics_lines)

    # Close files ######################################################################################################
    file_test.close()
    file_parsed.close()
    file_reads.close()
    file_statistics.close()

    return metrics_counter


########################################################################################################################

def statistical_analysis_with_known_genes_consecutive_frequency(testing_path, statistics_path, num_lines_for_read,
                                                                dataset_name_fastq="reads-both.fastq",
                                                                fusion_threshold=60, metrics_counter=MetricsCounter()):
    # Pre-processing ###################################################################################################
    # open files
    dataset_name = dataset_name_fastq.replace(".fastq", "")

    file_test = open(testing_path + 'test_fusion_result_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt')
    file_parsed = open(testing_path + 'parsed_gene_fusion_count_CFL_ICFL_COMB-30_K8_' + dataset_name + '.txt')
    file_reads = open(testing_path + dataset_name_fastq)

    # create new file
    file_statistics = open(statistics_path, 'w')

    # read lines
    test_lines = file_test.readlines()
    parsed_lines = file_parsed.readlines()
    reads_lines = file_reads.readlines()

    statistics_lines = []

    # Creazione dizionario
    file_genes = open('testing/RF_kfinger_clsf_report_CFL_ICFL_COMB-30_K8.csv')

    genes_lines = file_genes.readlines()
    genes_dictionary = {}
    for i in range(1, len(genes_lines)):
        line = genes_lines[i]
        line = line.split(',')

        value = line[0]
        key = i - 1

        genes_dictionary[key] = value

    file_genes.close()
    ####################################################################################################################

    # For each read create the lines in 'statistics_CFL_ICFL_COMB-30_K8.txt'
    for i in range(len(test_lines)):

        test_line = test_lines[i]
        parsed_line = parsed_lines[i]

        # Find corresponding read in 'dataset.fastq' ################################################################
        start_index = i * num_lines_for_read
        end_index = start_index + num_lines_for_read
        lines_for_current_read = reads_lines[start_index:end_index]
        read_line = lines_for_current_read[1]

        # Find most occurrent Gene_1 BEFORE break_fusion_index and Gene_2 AFTER break_fusion_index #####################

        # a) get list of kfinger predictions from test_line
        test_line_lst = test_line.split(' - PREDICTION: ')

        # b) splittiamo tra la prima parte contenente la sequenza di lunghezze e la seconda parte contenente le predictions
        kfingers_line_lst = test_line_lst[0]
        gene_line_lst = test_line_lst[1]

        # Extract the 2 reference fusion genes #########################################################################
        kfingers_line_lst = kfingers_line_lst.split('|')

        # Fusion Gene 1
        fusion_gene_1 = kfingers_line_lst[1]
        fusion_gene_1 = fusion_gene_1.split('.')
        fusion_gene_1 = fusion_gene_1[0]

        # Fusion Gene 2
        fusion_gene_2 = kfingers_line_lst[2]
        fusion_gene_2 = fusion_gene_2.split('.')
        fusion_gene_2 = fusion_gene_2[0]
        ################################################################################################################

        gene_line_lst = gene_line_lst.replace('[', '')
        gene_line_lst = gene_line_lst.replace(']', '')
        gene_line_lst = gene_line_lst.split()
        gene_line_lst = [genes_dictionary[int(g)] for g in gene_line_lst]

        # Add lines in 'statistics_CFL_ICFL_COMB-30_K8.txt'
        read_line = read_line.replace('\n', '')
        test_line = test_line.replace('\n', '')
        parsed_line = parsed_line.replace('\n', '')

        statistics_lines.append('READ: ' + read_line + '\n')
        statistics_lines.append('TEST_RESULT: ' + test_line + '\n')
        statistics_lines.append('SORTING GENES: ' + parsed_line + '\n')

        start_index_gene_1 = -1
        start_index_gene_2 = -1
        end_index_gene_1 = -1
        end_index_gene_2 = -1

        # Check relation between "gene 1 interval" and "gene 2 interval" ###############################################
        try:
            start_index_gene_1 = gene_line_lst.index(fusion_gene_1)
            end_index_gene_1 = len(gene_line_lst) - gene_line_lst[::-1].index(fusion_gene_1) - 1
            start_index_gene_2 = gene_line_lst.index(fusion_gene_2)
            end_index_gene_2 = len(gene_line_lst) - gene_line_lst[::-1].index(fusion_gene_2) - 1

        except ValueError:

            # stastistics_line = 'NO Break Fusion Interval!' + '\n'
            # stastistics_line = stastistics_line + '-\n'
            # stastistics_line = stastistics_line + '-\n\n'
            # statistics_lines.append(stastistics_line)

            # TN CASE
            # metrics_counter.increment_trueNegative(fusion_gene_1, fusion_gene_2)

            continue

        range_gene_1 = []
        range_gene_2 = []

        end_1 = 0
        if end_index_gene_1 == len(gene_line_lst):
            end_1 = end_index_gene_1
        else:
            end_1 = end_index_gene_1 + 1

        range_gene_1 = gene_line_lst[start_index_gene_1:end_1]
        range_gene_1 = smooth_range(lst=range_gene_1, threshold=1)

        end_2 = 0
        if end_index_gene_2 == len(gene_line_lst):
            end_2 = end_index_gene_2
        else:
            end_2 = end_index_gene_2 + 1
        range_gene_2 = gene_line_lst[start_index_gene_2:end_2]
        range_gene_2 = smooth_range(lst=range_gene_2, threshold=1)

        most_common_gene_in_range_gene_1 = most_consecutive_frequent(range_gene_1)
        most_common_gene_in_range_gene_2 = most_consecutive_frequent(range_gene_2)

        # Compute fusion_score
        perc_1 = ((end_1 - start_index_gene_1 + 1) * 100) / len(gene_line_lst)
        perc_2 = ((end_2 - start_index_gene_2 + 1) * 100) / len(gene_line_lst)
        perc_intersec = (len(gene_line_lst[start_index_gene_2:end_1]) * 100) / len(gene_line_lst)
        fusion_score = abs(100 - (perc_1 + perc_2 - perc_intersec))

        if fusion_score >= fusion_threshold:  # IL MODELLO DICE CHE E' CHIMERICO

            if fusion_gene_1 == fusion_gene_2:

                stastistics_line = 'NO Break Fusion!- fusion_score:' + str(fusion_score) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                    start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_2:' + str(fusion_gene_2) + ' |start_index:' + str(
                    start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                # stastistics_line = stastistics_line + '-\n'
                # stastistics_line = stastistics_line + '-\n\n'
                statistics_lines.append(stastistics_line)

                metrics_counter.increment_falsePositive(fusion_gene_1, fusion_gene_2)


            elif most_common_gene_in_range_gene_1 == fusion_gene_1 and most_common_gene_in_range_gene_2 == fusion_gene_2:

                stastistics_line = 'YES Break Fusion - fusion_score:' + str(fusion_score) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                    start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_2:' + str(fusion_gene_2) + ' |start_index:' + str(
                    start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                statistics_lines.append(stastistics_line)

                metrics_counter.increment_truePositive(fusion_gene_1, fusion_gene_2)

        elif fusion_score < fusion_threshold:

            if fusion_gene_1 == fusion_gene_2:

                stastistics_line = 'NO Break Fusion! - fusion_score:' + str(fusion_score) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                    start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_2:' + str(fusion_gene_2) + ' |start_index:' + str(
                    start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                # stastistics_line = stastistics_line + '-\n'
                # stastistics_line = stastistics_line + '-\n\n'
                statistics_lines.append(stastistics_line)

                metrics_counter.increment_trueNegative(fusion_gene_1, fusion_gene_2)


            elif most_common_gene_in_range_gene_1 == fusion_gene_1 and most_common_gene_in_range_gene_2 == fusion_gene_2:

                stastistics_line = 'YES Break Fusion - fusion_score:' + str(fusion_score) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_1:' + str(fusion_gene_1) + '|start_index:' + str(
                    start_index_gene_1) + '|end_index:' + str(end_index_gene_1) + '\n'
                stastistics_line = stastistics_line + 'fusion_gene_2:' + str(fusion_gene_2) + ' |start_index:' + str(
                    start_index_gene_2) + '|end_index:' + str(end_index_gene_2) + '\n\n'
                statistics_lines.append(stastistics_line)

                metrics_counter.increment_falseNegative(fusion_gene_1, fusion_gene_2)

        ################################################################################################################

        ################################################################################################################

    file_statistics.writelines(statistics_lines)

    # Close files ######################################################################################################
    file_test.close()
    file_parsed.close()
    file_reads.close()
    file_statistics.close()

    return metrics_counter


########################################################################################################################

def smooth_range(lst=[], threshold=1):
    target_gene = lst[0]
    target_gene_positions = [i for i, val in enumerate(lst) if val == target_gene]

    # scorro left-to-right fino a che non trovo 2 elementi consecutivi <= threshold
    start_index = 0
    for i in range(1, len(target_gene_positions)):

        if target_gene_positions[i] - target_gene_positions[i - 1] <= threshold:
            start_index = target_gene_positions[i - 1]
            break

    # scorro right-to-left fino a che non trovo 2 elementi consecutivi <= threshold
    end_index = len(target_gene_positions) - 1
    for i in range(len(target_gene_positions) - 2, 0, -1):

        if target_gene_positions[i + 1] - target_gene_positions[i] <= threshold:
            end_index = target_gene_positions[i + 1]
            break

    if end_index == len(target_gene_positions) - 1:
        lst = lst[start_index:len(target_gene_positions)]
    else:
        lst = lst[start_index:end_index + 1]

    return lst


def compute_fusion_accuracy(testing_path, path, statistical_name):
    # file_statistics = open('testing/statistics_CFL_ICFL_COMB-30_K8_dataset.txt')

    # Elimina il file csv precedente
    csv_file_path = testing_path + 'fusion_accuracy_' + statistical_name + '.csv'

    if os.path.exists(csv_file_path):

        # Apre il file CSV in modalità lettura
        with open(csv_file_path, 'r') as file:
            # Usa il modulo csv per creare un lettore CSV
            csv_reader = csv.reader(file)

            # Usa la funzione len() per ottenere il numero di righe
            number_of_rows = len(list(csv_reader))

            # Verifica se il numero di righe è maggiore di 2
            if number_of_rows > 2:
                # Riavvolge il file all'inizio
                file.seek(0)

                # Apre il file CSV in modalità scrittura
                with open(csv_file_path, 'w', newline='') as write_file:
                    # Crea un writer CSV
                    csv_writer = csv.writer(write_file)

                    # Sovrascrivi il file con una riga vuota
                    csv_writer.writerow([])

    file_statistics = open(path)

    statistics_lines = file_statistics.readlines()

    count_items = 0
    count_fusion = 0

    for line in statistics_lines:
        if line.find('Break Fusion') != -1:
            count_items += 1

            if line.find('YES') != -1:
                count_fusion += 1

    # control if count_items != 0 (Division by zero error)
    if count_items != 0:
        fusion_accuracy = (count_fusion * 100) / count_items
    else:
        fusion_accuracy = 0

    print('fusion accuracy : {}'.format(fusion_accuracy))

    # with open(testing_path + 'fusion_accuracy.txt', 'a') as file:
    #    file.write(f"{fusion_accuracy}\n")

    # Percorso del file CSV
    csv_file_path = testing_path + 'fusion_accuracy_' + statistical_name + '.csv'

    # Trova l'indice di "statistics_" nella stringa del percorso
    indice_statistics = path.find("statistics_")
    # Estrai la parte del percorso prima di "statistics_"
    path = path[:indice_statistics]

    # Scrivi i dati nel file CSV
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Scrivi l'intestazione se il file è vuoto
        if csv_file.tell() == 0:
            csv_writer.writerow(['Path', 'Accuracy'])

        # Scrivi i dati nel file CSV
        csv_writer.writerow([path, fusion_accuracy])

    print("Dati scritti con successo nel file.")

    file_statistics.close()


def compute_fusion_accuracy_and_statistics(args, num_lines_for_read):
    testing_path_chimeric = args.path1
    testing_path_non_chimeric = args.path2

    threshold = 0

    dataset_name_chimeric_fastq = args.fasta1
    dataset_name_chimeric = dataset_name_chimeric_fastq.replace(".fastq", "")

    dataset_name_non_chimeric_fastq = args.fasta2
    dataset_name_non_chimeric = dataset_name_non_chimeric_fastq.replace(".fastq", "")

    # 1) COUNT: genera file contenente la lista dei geni ordinati per numero di occorrenze
    gene_fusion_count(testing_path_chimeric, 'test_fusion_result_CFL_ICFL_COMB-30_K8_' + dataset_name_chimeric + '.txt',
                      'CFL_ICFL_COMB-30_K8', dataset_name_chimeric_fastq)

    gene_fusion_count(testing_path_non_chimeric,
                      'test_fusion_result_CFL_ICFL_COMB-30_K8_' + dataset_name_non_chimeric + '.txt',
                      'CFL_ICFL_COMB-30_K8', dataset_name_non_chimeric_fastq)

    # 2) PARSE: prende il file generato al passo precedente e sostituisce gli id numerici dei geni con le label
    parse_gene_fusion_result(testing_path_chimeric, dataset_name_chimeric_fastq)
    parse_gene_fusion_result(testing_path_non_chimeric, dataset_name_non_chimeric_fastq)

    # 3) ANALYZE: calcoliamo la % di reads che hanno i due geni di fusione tra i primi 3 elementi
    analyze_gene_fusion(testing_path_chimeric, dataset_name_chimeric_fastq)
    analyze_gene_fusion(testing_path_non_chimeric, dataset_name_non_chimeric_fastq)

    # 4) Statistical Analysis using the "break fusion index"
    #   - Per ogni read abbiamo 4 righe:
    #       a) riga 0: read corrispondente (in 'reads-both.fastq')
    #       a) riga 1: riga corrispondente nel file "test.........txt"
    #       b) riga 2: riga corrispondente nel file "parsed.........txt"
    #       c) riga 3: dice il punto di break (come posizione o numero kfinger) '$' ID_gene + classificato prima del break '$' ID_gene + classificato dopo il break
    # statistical_analysis_with_break_index(testing_path_chimeric, num_lines_for_read, dataset_name_chimeric_fastq)
    # statistical_analysis_with_break_index(testing_path_non_chimeric, num_lines_for_read,dataset_name_non_chimeric_fastq)
    # 5) Statistical Analysis using the "2 known genes"
    #   - Per ogni read abbiamo 4 righe:
    #       a) riga 0: read corrispondente (in 'reads-both.fastq')
    #       a) riga 1: riga corrispondente nel file "test.........txt"
    #       b) riga 2: riga corrispondente nel file "parsed.........txt"
    #       c) riga 3: dice il punto di break (come posizione o numero kfinger) '$' ID_gene + classificato prima del break '$' ID_gene + classificato dopo il break

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # statistical_analysis_with_known_genes_check_range_majority
    perform_statistical_analysis(statistical_analysis_with_known_genes_check_range_majority, testing_path_chimeric,
                                 testing_path_non_chimeric, dataset_name_chimeric, dataset_name_non_chimeric,
                                 num_lines_for_read, dataset_name_chimeric_fastq,
                                 dataset_name_non_chimeric_fastq, threshold)
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # statistical_analysis_with_known_genes_no_check_range_majority
    perform_statistical_analysis(statistical_analysis_with_known_genes_no_check_range_majority, testing_path_chimeric,
                                 testing_path_non_chimeric, dataset_name_chimeric, dataset_name_non_chimeric,
                                 num_lines_for_read, dataset_name_chimeric_fastq,
                                 dataset_name_non_chimeric_fastq, threshold)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # statistical_analysis_with_known_genes_consecutive_frequency

    perform_statistical_analysis(statistical_analysis_with_known_genes_consecutive_frequency, testing_path_chimeric,
                                 testing_path_non_chimeric, dataset_name_chimeric, dataset_name_non_chimeric,
                                 num_lines_for_read, dataset_name_chimeric_fastq,
                                 dataset_name_non_chimeric_fastq, threshold)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# SEARCH OPTIMAL THRESHOLD AND PERFORM STATISTICAL ANALYSIS( known_genes_consecutive_frequency or other)
def perform_statistical_analysis(function, testing_path_chimeric, testing_path_non_chimeric,
                                 dataset_name_chimeric, dataset_name_non_chimeric,
                                 num_lines_for_read, dataset_name_chimeric_fastq,
                                 dataset_name_non_chimeric_fastq, threshold):
    """
    Perform statistical analysis with known genes for a specific method.

    Parameters:
    - function: The statistical analysis function to be applied.
    - testing_path_chimeric: Path for chimeric data.
    - testing_path_non_chimeric: Path for non-chimeric data.
    - dataset_name_chimeric: Dataset name for chimeric data.
    - dataset_name_non_chimeric: Dataset name for non-chimeric data.
    - num_lines_for_read: Number of lines for reading data.
    - dataset_name_chimeric_fastq: Dataset name for chimeric FASTQ data.
    - dataset_name_non_chimeric_fastq: Dataset name for non-chimeric FASTQ data.
    - thresold: Threshold value.
    """
    # Reset metrics for statistical method
    metrics_counter = MetricsCounter()

    function_name = str(function.__name__).replace("statistical_", "")

    num_min_chimeric = 500

    threshold_search_range = (0.1, 5)
    threshold_search_step = 0.1
    target_f1_score = 0.90
    #
    # # Search in range threshold
    optimal_threshold = search_range_threshold(function, function_name, testing_path_chimeric,
                                               testing_path_non_chimeric,
                                               dataset_name_chimeric, dataset_name_non_chimeric,
                                               num_lines_for_read, dataset_name_chimeric_fastq,
                                               dataset_name_non_chimeric_fastq, target_f1_score,
                                               threshold_search_range, threshold_search_step)

    if optimal_threshold is not None:
        print(f"Optimal Threshold found: {optimal_threshold}")
    else:
        optimal_threshold = 1
        print("Optimal Threshold not found. Using default.")

    # Apply statistical analysis function to chimeric data
    function(testing_path_chimeric, testing_path_chimeric + f'statistics_{function_name}_{dataset_name_chimeric}.txt',
             num_lines_for_read, dataset_name_chimeric_fastq, optimal_threshold, metrics_counter)

    # Apply statistical analysis function to non-chimeric data
    function(testing_path_non_chimeric,
             testing_path_non_chimeric + f'statistics_{function_name}_{dataset_name_non_chimeric}.txt',
             num_lines_for_read, dataset_name_non_chimeric_fastq, optimal_threshold, metrics_counter)

    # Print results if available
    # metrics_counter.print_num_chimeric_nonChimeric()

    metrics_out = metrics_counter.calculate_metrics()
    print(metrics_out)

    metrics_counter.save_csv_metric("testing/" + "metrics_statistics_" + function_name + ".csv")
    metrics_counter.print_raw_metrics()

    compute_fusion_accuracy("testing/",
                            testing_path_chimeric + f'statistics_{function_name}_{dataset_name_chimeric}.txt',
                            f'statistics_{function_name}')
    compute_fusion_accuracy("testing/",
                            testing_path_non_chimeric + f'statistics_{function_name}_{dataset_name_non_chimeric}.txt',
                            f'statistics_{function_name}')
    print("\n")


# SEARCH OPTIMAL THRESHOLD FOR DATASET (CHIMERIC + NON CHIMERIC) FOR OPTIMAL METRICS
def search_range_threshold(function, function_name, testing_path_chimeric, testing_path_non_chimeric,
                           dataset_name_chimeric, dataset_name_non_chimeric,
                           num_lines_for_read, dataset_name_chimeric_fastq,
                           dataset_name_non_chimeric_fastq, target_f1_score,
                           threshold_search_range, threshold_search_step):
    """
    Search for the optimal threshold using an exhaustive search.

    Parameters:
    - function: The statistical analysis function to be applied.
    - testing_path_chimeric: Path for chimeric data.
    - testing_path_non_chimeric: Path for non-chimeric data.
    - dataset_name_chimeric: Dataset name for chimeric data.
    - dataset_name_non_chimeric: Dataset name for non-chimeric data.
    - num_lines_for_read: Number of lines for reading data.
    - dataset_name_chimeric_fastq: Dataset name for chimeric FASTQ data.
    - dataset_name_non_chimeric_fastq: Dataset name for non-chimeric FASTQ data.
    - target_f1_score: Target F1-score for finding optimal threshold.
    - threshold_search_range: Range for searching threshold (tuple, e.g., (0, 100)).
    - threshold_search_step: Step size for threshold search.
    """

    # Percorso del file di log
    log_file_path = 'logfile_' + function_name + '.log'

    # Verifica se il file di log esiste e, se sì, eliminalo
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # Logger configuration

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    best_f1_score = 0
    flag_target_f1_score = False
    optimal_threshold = None
    optimal_metrics = None

    # threshold_search_range[0] = START_THRSHOLD    threshold_search_range[1] = END_THRESHOLD
    current_threshold = threshold_search_range[0]
    while current_threshold <= threshold_search_range[1]:

        # print("current_threshold: ", current_threshold)
        # Reset metrics for each threshold
        metrics_counter = MetricsCounter()

        # Apply statistical analysis function to chimeric data with current threshold
        function(testing_path_chimeric,
                 testing_path_chimeric + f'statistics_{function_name}_{dataset_name_chimeric}.txt',
                 num_lines_for_read, dataset_name_chimeric_fastq, current_threshold, metrics_counter)

        # Apply statistical analysis function to non-chimeric data with current threshold
        function(testing_path_non_chimeric,
                 testing_path_non_chimeric + f'statistics_{function_name}_{dataset_name_non_chimeric}.txt',
                 num_lines_for_read, dataset_name_non_chimeric_fastq, current_threshold, metrics_counter)

        # Calculate F1-score
        metrics_out = metrics_counter.calculate_metrics()

        # print(metrics_out)

        current_f1_score = metrics_counter.f1_score

        # metrics_counter.print_raw_metrics()

        # Registra le informazioni nel file di log
        logging.info(f"Current Threshold: [{current_threshold}]")
        logging.info(metrics_out)
        logging.info(
            f"Metrics: TP: {metrics_counter.tp} - FP: {metrics_counter.fp} - TN: {metrics_counter.tn} - FN: {metrics_counter.fn}")

        logging.info(f"CHIMERIC: {metrics_counter.chimeric} - NON CHIMERIC: {metrics_counter.non_chimeric}\n\n")

        # Se f1 score attuale è migliore o uguale di quello previsto e se non ha mai superato il target(flag = false)
        if current_f1_score >= target_f1_score and flag_target_f1_score == False:
            best_f1_score = current_f1_score
            optimal_threshold = current_threshold
            optimal_metrics = metrics_counter
            flag_target_f1_score = True

        # Se il flag = true, allora controllo solo che l'f1 score attuale sia migliore dell'f1 score ottimale attuale
        elif current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            optimal_threshold = current_threshold
            optimal_metrics = metrics_counter

        elif current_f1_score == best_f1_score:
            if metrics_counter.tp > optimal_metrics.tp:
                best_f1_score = current_f1_score
                optimal_threshold = current_threshold
                optimal_metrics = metrics_counter

            elif metrics_counter.fp < optimal_metrics.fp:
                best_f1_score = current_f1_score
                optimal_threshold = current_threshold
                optimal_metrics = metrics_counter
            elif metrics_counter.fn < optimal_metrics.fn:
                best_f1_score = current_f1_score
                optimal_threshold = current_threshold
                optimal_metrics = metrics_counter

        if current_threshold < threshold_search_range[1]:
            current_threshold += threshold_search_step
        else:
            logging.info(f"OPTIMAL THRESHOLD: [{optimal_threshold}]")
            return optimal_threshold

    logging.info(f"OPTIMAL THRESHOLD: [{optimal_threshold}]")

    return optimal_threshold


def compute_only_fusion_accuracy(args, num_lines_for_read):
    testing_path_chimeric = args.path
    testing_path_non_chimeric = args.path1

    dataset_name_chimeric_fastq = args.fasta
    dataset_name_chimeric = dataset_name_chimeric_fastq.replace(".fastq", "")

    dataset_name_non_chimeric_fastq = args.fasta1
    dataset_name_non_chimeric = dataset_name_non_chimeric_fastq.replace(".fastq", "")
    '''
    compute_fusion_accuracy('testing/',
                            testing_path_chimeric + 'statistics_check_range_CFL_ICFL_COMB-30_K8_' + dataset_name_chimeric + '.txt')
    compute_fusion_accuracy('testing/',
                            testing_path_chimeric + 'statistics_no_check_range_CFL_ICFL_COMB-30_K8_' + dataset_name_chimeric + '.txt')
    compute_fusion_accuracy('testing/',
                            testing_path_chimeric + 'statistics_consecutive_frequency_CFL_ICFL_COMB-30_K8_' + dataset_name_chimeric + '.txt')

    compute_fusion_accuracy('testing/',
                            testing_path_non_chimeric + 'statistics_check_range_CFL_ICFL_COMB-30_K8_' + dataset_name_non_chimeric + '.txt')
    compute_fusion_accuracy('testing/',
                            testing_path_non_chimeric + 'statistics_no_check_range_CFL_ICFL_COMB-30_K8_' + dataset_name_non_chimeric + '.txt')
    compute_fusion_accuracy('testing/',
                            testing_path_non_chimeric + 'statistics_consecutive_frequency_CFL_ICFL_COMB-30_K8_' + dataset_name_non_chimeric + '.txt')
    '''


##################################################### MAIN #############################################################
########################################################################################################################
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Gestione argomenti ###############################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', dest='step', action='store', default='1rf')
    parser.add_argument('--path', dest='path', action='store', default='testing/')
    parser.add_argument('--path1', dest='path1', action='store', default='testing/chimeric')
    parser.add_argument('--path2', dest='path2', action='store', default='testing/non_chimeric')
    parser.add_argument('--fasta1', dest='fasta1', action='store', default='sample_10M_genes.fastq.gz')
    parser.add_argument('--fasta2', dest='fasta2', action='store', default='sample_10M_genes.fastq.gz')
    parser.add_argument('--result_file', dest='result_file', action='store',
                        default='test_fusion_result_CFL_ICFL_COMB-30_K8.txt')
    parser.add_argument('--fact', dest='fact', action='store', default='no_create')
    parser.add_argument('--shift', dest='shift', action='store', default='no_shift')
    parser.add_argument('--best_model', dest='best_model', action='store', default='RF_CFL_ICFL-20_K8.pickle')
    parser.add_argument('--k_type', dest='k_type', action='store', default='extended')
    parser.add_argument('--k_value', dest='k_value', action='store', default=3, type=int)
    parser.add_argument('--filter', dest='filter', action='store', default='list')
    parser.add_argument('--random', dest='random', action='store', default='no_random')
    parser.add_argument('--type_factorization', dest='type_factorization', action='store', default='CFL')
    parser.add_argument('--dictionary', dest='dictionary', action='store', default='no')
    parser.add_argument('-n', dest='n', action='store', default=1, type=int)

    args = parser.parse_args()

    # prova_testing_reads_RF_fingerprint_mp_step('testing/', 'example_sample_10M_genes.fastq.gz', 'list', 'ICFL_COMB', 'no_create', 'no_shift', 'RF_fingerprint_classifier_ICFL_COMB.pickle')

    if args.step == 'test_fusion':
        print('\nTesting Step: TEST set of READS with FUSION on Best k-finger classification...\n')
        dataset_path_chimeric = args.path1
        dataset_name_chimeric = args.fasta1

        dataset_path_non_chimeric = args.path2
        dataset_name_non_chimeric = args.fasta2
        testing_reads_fusion_mp_step(args, dataset_path_chimeric, dataset_name_chimeric)
        testing_reads_fusion_mp_step(args, dataset_path_non_chimeric, dataset_name_non_chimeric)
        # dataset_name_fastq = args.fasta

    if args.step == 'test_result':
        print("\nCreate Statistics file and compute fusion score")
        compute_fusion_accuracy_and_statistics(args, 2)

    if args.step == 'compute_only_result':
        # Compute only fusion accuracy
        compute_only_fusion_accuracy(args, 2)
