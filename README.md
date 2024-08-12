# Combinatorics Machine Learning for *Gene Fusion* Detection 
--------------------------------------------------------------------------------------------------------------------

In the field of computational biology, the accurate mapping of RNA sequence reads (RNA-Seq) to their respective origin genes is a fundamental and prerequisite goal for studying gene fusions. Gene fusions, representing a mechanism of chromosomal rearrangement where two (or more) genes merge into a single gene (fusion gene), are often associated with cancer. Chromosomal rearrangements leading to gene fusions are particularly prevalent in sarcomas and hematopoietic neoplasms and are also common in solid tumors.

Combinatorics-ML-Gene-Fusion is an ambitious bioinformatics project designed to detect gene fusions of two genes within a single transcript. It explores a hybrid approach between the contexts of *Machine Learning* and *Combinatorics on Words*, relying on an efficient system of factorizations. The representations are expressed by *k-fingers*, which are k-mers extracted from a gene's *fingerprint* or signature. 

**GOALs of the project**:
1. to develop advanced ML-based strategies for the gene fusion identification exploiting the CW-based embeddings for biological sequences
2. to carry out experiments to assess the effectiveness of  model for classify reads as chimeric or not chimeric.
  
We define *gene fusion* as a chromosomal rearrangement that combines two genes into a single fusion gene, resulting in a chimeric transcript formed by the concatenation of segments from each of the original genes.

Our experiments is based on the files into the *dataset* folder organized in: *dataset_chimeric* that contains the chimeric sequences and *dataset_no_chimeric* that contains sequences of no fused transcripts.

The classification methods developed can be catgorized as follow:
1) **MLE based**. These methods are based on the idea that each sequencing read is represented by the list of 𝑘-fingers extracted by a *superimposed fingerprint*, and consists in the following 4 steps:
   
   1) Generation of fingerprints/k-fingers by factorizing transcripts referring to genes from an arbitrarily gene panel.

   2) Training a model using a training dataset consisting of all labeled k-fingers with the origin gene. This allows us to assess their repetitiveness within the gene itself.

   3) Classification of chimeric and non-chimeric reads, starting from the predictions of point 2 and using *combinatorial calculations* (criterion) to evaluate how two genes are expressed within a transcript.

   4) After assessing the expression levels of the two genes, calculating a fusion score to determine whether the read is chimeric (fusion of two genes) or non-chimeric (no gene fusion).

    Thanks to the generated results, we were able to measure with a certain precision how chimeric or non-chimeric a dataset was by adaptively thresholding it. This allowed us to identify the precise point in the dataset from which to extract as many chimeric reads as possible with an optimal compromise, resulting in a minimal decrease in metrics.  
    
    We defined 4 variants, each one corresponding to a specific criterion:
      - **MLE_check** (MLE with Check interval criterion),
      - **MLE_no_check** (MLE with No Check interval criterion),
      - **MLE_repetitive** (MLE with Repetitive criterion),
      - **MLE_ensemble** (MLE with Ensemble criterion);

2) **MGE based**. Methods based on the idea that each read 𝑤 is represented by a *De Bruijn hypergraph* and built on the set of all the 𝑘-fingers extracted by a set of superimposed fingerprints for 𝑤
   We definied 2 variants, *basic* and *generalized*:
    - **MGE_basic** (MGE with basic experiment): This method uses *Graph Convolutional Network* (GCN)
    - **MGE_generalized** (MGE with generalizedexperiment): This methos uses *Hypergraph Convolutional Network* (HGCN)


--------------------------------------------------------------------------------------------------------------------
## Istruction for MLE based experiments

Install the libraries with the following command:
   
      ```pip install -r requirements.txt```
 
**information about options for various commands**:

- **dictionary= yes or no**: you can decide whether to use fingerprint management for each 300 window of each transcript with adaptive split (yes) or without adaptive split (no)
  
- **k_value = 3 to 8 (8 best value)**: is the number of k-mers extracted from a fingerprint (k-finger) and it has been found that the value with the best results is (k_value = 8)
  
- **n = 4 or 8**: is the number of processors used for parallel execution, for an averagely powerful machine the value (n = 4) is recommended, otherwise for more powerful machines or servers (n = 8)
    
    - if k = 4 would give an error like this: **UserWarning: Loky-backed parallel loops cannot be called in a multiprocessing,, setting n_jobs=1n_jobs = min(effective_n_jobs(n_jobs), n_estimators)**
      or other types of problems, **set (k = 2)**

- **type_factorization**: ```CFL```, ```ICFL```, ```CFL_ICFL-10```, ```CFL_ICFL-20```, ```CFL_ICFL-30```, ```CFL_COMB```, ```ICFL_COMB```, ```CFL_ICFL_COMB-10```, ```CFL_ICFL_COMB-20```, ```CFL_ICFL_COMB-30```
  
--------------------------------------------------------------------------------------------------------------------

**Instruction to train the models (With dictionary)**


1) **Generate Fingerprint**
   
   COMMAND TO GENERATE FINGERPRINTS FROM GENE TRANSCRIPTS (ENSG + SEQUENCE) (**transcripts.fa**)   


   ```python fingerprint.py --type 1f_np --path training/ --fasta transcripts.fa --type_factorization CFL_ICFL_COMB-30 -n 4 --dictionary yes```


2) **Training Model**

   1) **Generate dataset X and Dataset Y**

      COMMAND TO GENERATE DATASET_X AND DATASET_Y FROM FINGERPRINT DATA AND FACTORIZATIONS

      ```python training.py --step dataset --path training/ --type_factorization CFL_ICFL_COMB-30 --k_value 8```

   2) **Train Model**
      
      COMMAND TO TRAIN THE MODEL FROM DATASET_X AND DATASET_Y

       ```python training.py --step train --path training/ --type_factorization CFL_ICFL_COMB-30 --k_value 8 --model RF -n 4```

   3) **Important steps**

      1) MOVE (MODEL) ```RF_type_factorization_K_value.pickle``` (example: ```RF_CFL_ICFL_COMB-30_K8.pickle```) from **src/training** to **Csrc/testing**
  
      2) MOVE (REPORT MODEL)  ```RF_kfinger_clsf_report_type_factorization_K_value.csv``` (example: ```RF_kfinger_clsf_report_CFL_ICFL_COMB-30_K8.csv```) from **src/training** to **src/testing**
      
      3) MOVE (DICTIONARY) ```dictionary_type_factorization.txt``` (example: ```dictionary_CFL_ICFL_COMB-30.txt```) from **src/training** to **src/testing**
     
3) **Test Dataset generated with fusim**


   **information about options for Testing commands**:

   - fasta1 = dataset_name_chimeric.fastq in **testing/chimeric/**
   - fasta2 = dataset_name_non_chimeric.fastq in **testing/non_chimeric/**
   - best_model = ```RF_type_factorization_K_value.pickle``` MOVED to the previous step
   - if you have problems with n = 4, use n = 2

       
   1) COMMAND TO GENERATE THE TEST_RESULT_FUSION FILE
       - ```python testing.py --step test_fusion --path testing/ --path1 testing/chimeric/ --path2 testing/non_chimeric/ --fasta1 dataset_chimeric1.fastq --fasta2 dataset_non_chimeric1.fastq  --best_model RF_CFL_ICFL_COMB-30_K8.pickle --type_factorization CFL_ICFL_COMB-30 --k_value 8 -n 2 --dictionary yes```


 
   2) COMMAND THAT USES TEST RESULT_FUSION AND GENERATES FUSION SCORE AND STATISTICS FILES FOR DATASET_CHIMERIC AND DATASET_NON_CHIMERIC
       - ```python testing.py --step test_result --path1 testing/chimeric/ --path2 testing/non_chimeric/ --fasta1 dataset_chimeric1.fastq --fasta2 dataset_non_chimeric1.fastq  --best_model RF_CFL_ICFL_COMB-30_K8.pickle --type_factorization CFL_ICFL_COMB-30 --k_value 8 -n 4 --dictionary yes```

--------------------------------------------------------------------------------------------------------------------  
**Instruction to train the models (Without dictionary)**

  1) **Generate Fingerprint**
   
     COMMAND TO GENERATE FINGERPRINTS FROM GENE TRANSCRIPTS (ENSG + SEQUENCE) (**transcripts.fa**)   


     ```python fingerprint.py --type 1f_np --path training/ --fasta transcripts.fa --type_factorization CFL_ICFL_COMB-30 --fact create --shift shift -n 4```


2) **Training Model**

   1) **Generate dataset X and Dataset Y**

      COMMAND TO GENERATE DATASET_X AND DATASET_Y FROM FINGERPRINT DATA AND FACTORIZATIONS

      ```python training.py --step train --path training/ --type_factorization CFL_ICFL_COMB-30 --k_value 8 --model RF -n 4```

   2) **Train Model**
      
      COMMAND TO TRAIN THE MODEL FROM DATASET_X AND DATASET_Y

       ```python training.py --step train --path training/ --type_factorization CFL_ICFL_COMB-30 --k_value 8 --model RF -n 4```

   3) **Important steps**

      1) MOVE (MODEL) ```RF_type_factorization_K_value.pickle``` (example: ```RF_CFL_ICFL_COMB-30_K8.pickle```) from **src/training** to **src/testing**
  
      2) MOVE (REPORT MODEL)  ```RF_kfinger_clsf_report_type_factorization_K_value.csv``` (example: ```RF_kfinger_clsf_report_CFL_ICFL_COMB-30_K8.csv```) from **src/training** to **src/testing**
      
      3) MOVE (DICTIONARY) ```dictionary_type_factorization.txt``` (example: ```dictionary_CFL_ICFL_COMB-30.txt```) from **src/training** to **src/testing**
 
     
3) **Test Dataset generated with fusim**


   **information about options for Testing commands**:

   - fasta1 = dataset_name_chimeric.fastq in **testing/chimeric/**
   - fasta2 = dataset_name_non_chimeric.fastq in **testing/non_chimeric/**
   - best_model = ```RF_type_factorization_K_value.pickle``` MOVED to the previous step
   - if you have problems with n = 4, use n = 2

     
   1) COMMAND TO GENERATE THE TEST_RESULT_FUSION FILE
       - ```python testing.py --step test_fusion --path testing/ --path1 testing/chimeric/ --path2 testing/non_chimeric/ --fasta1 dataset_chimeric1.fastq --fasta2 dataset_non_chimeric1.fastq  --best_model RF_CFL_ICFL_COMB-30_K8.pickle --type_factorization CFL_ICFL_COMB-30 --k_value 8 -n 2```


 
   2) COMMAND THAT USES TEST RESULT_FUSION AND GENERATES FUSION SCORE AND STATISTICS FILES FOR DATASET_CHIMERIC AND DATASET_NON_CHIMERIC
       - ```python testing.py --step test_result --path1 testing/chimeric/ --path2 testing/non_chimeric/ --fasta1 dataset_chimeric1.fastq --fasta2 dataset_non_chimeric1.fastq  --best_model RF_CFL_ICFL_COMB-30_K8.pickle --type_factorization CFL_ICFL_COMB-30 --k_value 8 -n 4```

-------------------------------------------------
##MGE and MML Experiments
To run the MGE and the MML experiments, the .ipynb file is formated in order to run the experiment in succession.
The experimenst takes in input the fingeprints files generable from the previous explained procedure.

Install the libraries with the following command:
   
      ```pip install -r requirements_MGE_MLL.txt```
To run the MGE experiments:
      ```python MGE_experiment.py --balanced_bool <balance_flag> --filename_fuse <path_to_fuse_file> --filename_no_fuse <path_to_no_fuse_file> --k <k_value>```
      example:
      ```python MGE_experiment.py --balanced_bool False --filename_fuse /path/to/fingerprint_fuse.txt --filename_no_fuse /path/to/fingerprint_no_fuse.txt --k 4```





