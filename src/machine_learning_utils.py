import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from fingerprint_utils import computeWindow


# Mapping pool for training in multiprocessing
def mapping_pool_train(path="fingerprint/ML/", tuple_fact_k=('CFL',3, 'RF')):
    train(path, tuple_fact_k[0], tuple_fact_k[1], tuple_fact_k[2])

# Training of classifiers
def train(path="fingerprint/ML/", type_factorization='CFL', k=3, type_model = 'RF'):

    if type_model == 'RF':
        random_forest_kfinger(path=path, type_factorization=type_factorization, k=k)


# Split of the dataset
def train_test_generator(dataset_name):
    
    X = pickle.load(open(dataset_name,"rb"))
    y_dataset = dataset_name.replace("X", "y")
    y = pickle.load(open(y_dataset,"rb"))

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),)

    training_set_data,test_set_data,training_set_labels,test_set_labels = train_test_split(X,integer_encoded,test_size=0.7,stratify=integer_encoded)

    scaler = MinMaxScaler()
    train_scaled_D = scaler.fit_transform(training_set_data)
    test_scaled_D = scaler.transform(test_set_data)

    return (train_scaled_D,test_scaled_D,training_set_labels,test_set_labels, label_encoder,scaler)
    

# Compute the classification thresholds for each class gene
def compute_classification_thresholds(model=None, test_set=None, labels=None,clsf=None):

    # Probability predictions (matrix x_samples X n_genes)
    prediction_proba = model.predict_proba(test_set)
    prediction_proba = np.array(prediction_proba)

    # Real prediction (array 1 X n_genes)
    prediction = model.predict(test_set)
    prediction = prediction.tolist()

    thresholds = []

    # For each class
    for i in range(len(labels)):
        samples_predicted_for_i = []
        for j in range(len(prediction)):
            sample = prediction[j]
            if sample == i:
                samples_predicted_for_i.append((prediction_proba[j])[i])

        threshold = np.amin(samples_predicted_for_i, axis=0)
        thresholds = np.append(thresholds, [threshold])

    for lbl, threshold in zip(labels,thresholds):
        dict_item = clsf[lbl]
        new_dict={'precision':dict_item['precision'],'recall':dict_item['recall'],'f1-score':dict_item['f1-score'],'support':dict_item['support'],'threshold':threshold}
        clsf[lbl] = new_dict

    return clsf


# Random forest k_finger classifier
def random_forest_kfinger(path="training/", type_factorization='CFL', k=8):

    print('\nTrain RF k_finger classifier (%s, %s) - start...' % (type_factorization, k))

    # Create name dataset
    dataset_name = path + "dataset_X_" + type_factorization + "_K" + str(k) + ".pickle"
    train_scaled_D,test_scaled_D, training_set_labels, test_set_labels, label_encoder, min_max_scaler = train_test_generator(dataset_name)

    n_genes = len(set(training_set_labels))
    classificatore = RandomForestClassifier(n_estimators=8, min_samples_leaf=1, n_jobs=-1)
    classificatore.fit(train_scaled_D, training_set_labels)
    
    labels_originarie = label_encoder.inverse_transform(np.arange(n_genes))
    y_pred = classificatore.predict(test_scaled_D)

    clsf = classification_report(test_set_labels, y_pred, target_names =labels_originarie, output_dict=True)
    #clsf = compute_classification_thresholds(model=classificatore, test_set=test_scaled_D, labels=labels_originarie, clsf=clsf)
    clsf_report = pd.DataFrame(data=clsf).transpose()
    csv_name = path + "RF_kfinger_clsf_report_" + type_factorization + "_K" + str(k) + ".csv"
    clsf_report.to_csv(csv_name, index= True)

    #print("Random Forest accuracy: ", accuracy_score(test_set_labels,classificatore.predict(test_scaled_D)))

    # Pickle [RF model, labels_originarie,
    pickle.dump([classificatore,label_encoder, min_max_scaler], open(path + "RF_" + type_factorization + "_K" + str(k) + ".pickle", 'wb'))

    print('\nTrain RF k_finger classifier (%s, %s) - stop!' % (type_factorization, k))


# RULE-BASED READ CLASSIFIER
# Given a set of reads, performs classification by using the majority (or thresholds) criterion on  Best k-finger classification
def test_reads_fusion(list_best_model=None, path='testing/', type_factorization='CFL_ICFL-20', k_value=8, fingerprint_block = []):

    print('\nRule-based read classifier - start...')
    test_lines = []

    # best model
    best_model = list_best_model[0]
    best_label_encoder = list_best_model[1]
    best_min_max_scaler = list_best_model[2]

    for fingerprint in fingerprint_block[0]:

        #print(fingerprint)
        lengths_list = fingerprint.split()

        str_k_fingers = lengths_list[1:]
        #print(str_k_fingers)

        str_k_fingers = ''.join(str(fact) + '-' for fact in str_k_fingers)
        #print(str_k_fingers)

        str_k_fingers = str_k_fingers.replace('-',' ')
        #print(str_k_fingers)

        list_k_fingers = str_k_fingers.split('|')
        #print(list_k_fingers)

        str_pred = ''
        #for list_k in list_k_fingers:
        #for i in range(len(list_k_fingers) - 1):
        for i in range(len(list_k_fingers)):
            list_k = list_k_fingers[i]
            list_k = list_k.split()
            #print(list_k)

            #k_fingers = computeWindow(lengths_list[1:], k_value, k_window="extended")
            k_fingers = computeWindow(list_k, k_value, k_window="extended")

            # Scaler tests set
            test_scaled_D = best_min_max_scaler.transform(k_fingers)
            y_pred = best_model.predict(test_scaled_D)

            str_pred = str_pred + str(y_pred) + ' '
            str_pred = str_pred.replace('\n','')

        test_lines.append('FINGERPRINT: ' + fingerprint + ' - PREDICTION: ' + str_pred + '\n')

    print('\nRule-based read classifier - stop!')

    return test_lines