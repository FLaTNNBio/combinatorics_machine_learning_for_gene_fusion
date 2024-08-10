import argparse
import itertools

from fingerprint_utils import mapping_pool_create_ML_dataset
from multiprocessing.pool import Pool
from functools import partial
from machine_learning_utils import mapping_pool_train


# Create datasets (args.step = 'dataset') ##############################################################################
def experiment_dataset_step(args):

    task = [i for i in itertools.product([args.type_factorization], [args.k_value])]

    with Pool(args.n) as pool:
        func = partial(mapping_pool_create_ML_dataset, args.path, args.k_type, 'no_string')
        for _ in pool.map(func, task):
            print('Mapping create ML dataset')



# Train classifiers (args.step = 'train') ##############################################################################
def experiment_training_step(args):

    task = [i for i in itertools.product([args.type_factorization], [args.k_value], [args.model])]
    with Pool(args.n) as pool:
        func = partial(mapping_pool_train, args.path)
        for _ in pool.map(func, task):
            print('mapping train models')



########################################################################################################################
##################################################### MAIN #############################################################
########################################################################################################################
if __name__ == '__main__':

    # Gestione argomenti ###############################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', dest='step', action='store', default='fingerprint')
    parser.add_argument('--path', dest='path', action='store', default='training/')
    parser.add_argument('--type_factorization', dest='type_factorization', action='store', default='CFL')
    parser.add_argument('--model', dest='model', action='store', default='RF')
    parser.add_argument('--k_value', dest='k_value', action='store', default=3, type=int)
    parser.add_argument('-n', dest='n', action='store', default=1, type=int)
    parser.add_argument('--k_type', dest='k_type', action='store', default='extended')

    args = parser.parse_args()

    # BUILD DATASET
    if args.step == 'dataset':
        print('\nTraining Step: DATASET...\n')
        experiment_dataset_step(args)

    elif args.step == 'train':
        print('\nTraining Step: TRAINING...\n')
        experiment_training_step(args)

