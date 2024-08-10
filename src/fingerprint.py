import argparse
import pickle
from fingerprint_utils import extract_reads,compute_fingerprint_by_list
from multiprocessing.pool import Pool
from functools import partial
from factorizations import CFL, ICFL_recursive, CFL_icfl
from factorizations_comb import d_cfl, d_icfl, d_cfl_icfl


# Create fingerprint files (args.step = '1f_np') #################################################################
def experiment_fingerprint_1f_np_step(args):

    # Input FASTA file containing transcripts
    input_fasta = args.path + args.fasta

    # dictionary #######################################################################################################
    dictionary_file = None
    dictionary_lines = None
    if args.dictionary == 'yes':
        print('XXX')
        dictionary_file = open("%s" % args.path + "dictionary_" + args.type_factorization + ".txt", 'w')
        dictionary_lines = []
        ####################################################################################################################

    # Extract of reads (Format = ID GENE read)
    read_lines = extract_reads(name_file=input_fasta)

    if len(read_lines) == 0:
        print('No reads extracted!')
        exit(-1)

    print('\nCompute fingerprint by list (%s, %s) - start...' % (args.type_factorization, args.fact))

    fingerprint_file = open("%s" % args.path + "fingerprint_" + args.type_factorization + ".txt", 'w')
    fact_fingerprint_file = None
    if args.fact == 'create':
        # Create file containing factorizations
        fact_fingerprint_file = open("%s" % args.path + "fact_fingerprint_" + args.type_factorization + ".txt", 'w')

    # SPLIT for multiprocessing
    size = int(len(read_lines)/args.n)
    splitted_lines = [read_lines[i:i + size] for i in range(0, len(read_lines), size)]

    with Pool(args.n) as pool:

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

        func = partial(compute_fingerprint_by_list, args.fact, args.shift, factorization, T, args.dictionary)

        fingerprint_lines = []
        fingerprint_fact_lines = []
        for res in pool.map(func, splitted_lines):
            fingerprint_lines = fingerprint_lines + res[0]
            fingerprint_fact_lines = fingerprint_fact_lines + res[1]

            if args.dictionary == 'yes':
                dictionary_lines = dictionary_lines + res[2]

        fingerprint_file.writelines(fingerprint_lines)
        if args.fact == 'create':
            fact_fingerprint_file.writelines(fingerprint_fact_lines)

        fingerprint_file.close()

        if args.fact == 'create':
            fact_fingerprint_file.close()


        # dictionary ###################################################################################################
        if args.dictionary == 'yes':
            dictionary_lines = list(dict.fromkeys(dictionary_lines))

            for i in range(len(dictionary_lines) - 1):
                dictionary_file.write(dictionary_lines[i] + '\n')
            dictionary_file.write(dictionary_lines[len(dictionary_lines)-1])

            dictionary_file.close()

        ################################################################################################################
        print('\nCompute fingerprint by list (%s, %s) - stop!' % (args.type_factorization, args.fact))

##################################################### MAIN #############################################################
########################################################################################################################
if __name__ == '__main__':

    # Gestione argomenti ###############################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', dest='type', action='store', default='1f_np')
    parser.add_argument('--path', dest='path', action='store', default='training/')
    parser.add_argument('--type_factorization', dest='type_factorization', action='store',default='CFL')
    parser.add_argument('--fasta', dest='fasta', action='store', default='transcript_genes.fa')
    parser.add_argument('--fact', dest='fact', action='store', default='create')
    parser.add_argument('--shift', dest='shift', action='store', default='shift')
    parser.add_argument('--dictionary', dest='dictionary', action='store', default='no')
    parser.add_argument('-n', dest='n', action='store', default=1, type=int)

    args = parser.parse_args()

    if args.type == '1f_np':
        print('\nFingerprint Step: 1f_np...\n')
        experiment_fingerprint_1f_np_step(args)
