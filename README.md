# combinatorics_machine_learning_for_gene_fusion
--------------------------------------------------------------------------------------------------------------------
**ABSTRACT**

In the field of computational biology, the accurate mapping of RNA sequence reads (RNA-Seq) to their respective origin genes is a fundamental and prerequisite goal for studying gene fusions. Gene fusions, representing a mechanism of chromosomal rearrangement where two (or more) genes merge into a single gene (fusion gene), are often associated with cancer. Chromosomal rearrangements leading to gene fusions are particularly prevalent in sarcomas and hematopoietic neoplasms and are also common in solid tumors.

Combinatorics-ML-Gene-Fusion is an ambitious bioinformatics project designed to detect gene fusions of two genes within a single transcript. It explores a hybrid approach between the contexts of machine learning and combinatorial computation, relying on an efficient system of factorizations. The representations are expressed by k-fingers, which are k-mers extracted from a gene's fingerprint or signature. The project can be summarized in four main points:

1) Generation of fingerprints or k-fingers by factorizing transcripts referring to genes from an arbitrarily sized gene panel.

2) Training a model using a training dataset consisting of all labeled k-fingers with the origin gene. This allows us to assess their repetitiveness within the gene itself.

3) Classification of chimeric and non-chimeric reads, starting from the predictions of point 2 and using combinatorial calculations to evaluate how two genes are expressed within a transcript.

4) After assessing the expression levels of the two genes, calculating a fusion score to determine whether the read is chimeric (fusion of two genes) or non-chimeric (no gene fusion).

The programming language **Python**, along with the libraries **scikit-learn**, **numpy**, and **pandas**, was utilized in building the model.

Thanks to the generated results, we were able to measure with a certain precision how chimeric or non-chimeric a dataset was by adaptively thresholding it. This allowed us to identify the precise point in the dataset from which to extract as many chimeric reads as possible with an optimal compromise, resulting in a minimal decrease in metrics.  

 
GOAL:
1. to develop advanced ML-based strategies for the gene fusion identification exploiting the CW-based embeddings for biological sequences
2. to carry out experiments to assess the effectiveness of  model for classify reads as chimeric or not chimeric.
  
We define gene fusion as a chromosomal rearrangement that combines two genes into a single fusion gene, resulting in a chimeric transcript formed by the concatenation of segments from each of the original genes.

--------------------------------------------------------------------------------------------------------------------

Our experiments is based on the files into the *dataset* folder organized in: *dataset_chimeric* that contains the chimeric sequences and *dataset_no_chimeric* that contains seuqneces of no fused transcripts.

I modelli sono (descrizione + vedere tabella)
The models are (i) the 4 variants (one for each criterion) of the method
for list-embedding (MLE):
- **MLE_check** (MLE with Check interval criterion),
- **MLE_no_check** (MLE with No Check interval criterion), MLE_repetitive (MLE with Repetitive criterion),
- **MLE_ensemble** (MLE with Ensemble criterion);
(ii) the 2 variants (basic and generalized) of the method for graph-list embedding (MGE)
- **MGE_basic** (MGE with basic experiment),
- **MGE_generalized** (MGE with generalizedexperiment);
(iii) the “full” machine learning-based method (**MML**) proposed in.


--------------------------------------------------------------------------------------------------------------------
Parte eduardo:
requirements
information about options for various commands sano sano
Instructions for conducting a test -> Instruction to train the models (With dictionary)
Instructions for conducting a test (WITHOUT DICTIONARY) -> Instruction to train the models (Without dictionary)

Va chiedo ad eduardo come esplicitare come far correre, no check e gli altri nella tabella ( dare la pissibilta da comand line di far correre i test che si boglino)
-------------------------------------------------
Parte Dino
mettere sorgente in .py




