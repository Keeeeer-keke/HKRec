# Implementation of paper: Hyper-relational Knowledge Enhanced Network for Hypertension Medication Recommendation

### Folder Specification
- DATA_prepare/
    - patient_hyp.ipynb: Extract the EHR of patients with hypertension.
	- med_hyp.ipynb: Extract the antihypertensive medications used by patients with hypertension.
	- ndc2RXCUI.txt: NDC to xnorm mapping file
	- RXCUI2atc4.csv: NDC code to ATC4 level code mapping file
- DR_module/
	- data/ **(For a fair comparision, we use the same data and pre-processing scripts used in [Safedrug](https://github.com/ycq091044/SafeDrug))**
		- input/ **mapping files that collected from external sources**
			- PRESCRIPTIONS.csv: the prescription file from MIMIC-IV_2.2 dataset
			- DIAGNOSES_ICD.csv: the diagnosis file from MIMIC-IV_2.2 dataset
			- PROCEDURES_ICD.csv: the procedure file from MIMIC-IV_2.2 dataset
			- RXCUI2atc4.csv: xnorm code to ATC4 code mapping file. This file is obtained from https://github.com/sjy1203/GAMENet.
			- ndc2RXCUI.txt: NDC to xnorm mapping file. This file is obtained from https://github.com/sjy1203/GAMENet.
			- drug-atc.csv: drug to atc code mapping file. This file is obtained from https://github.com/sjy1203/GAMENet.
			- drug-DDI.csv: this a large file, could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
			- drugbank_drugs_info.csv: drug information table downloaded from drugbank here https://www.dropbox.com/s/angoirabxurjljh/drugbank_drugs_info.csv?dl=0, which is used to map drug name to drug SMILES string.
			- ICD_tensor.csv: This is the medical entity vector file obtained by training HKG through KG_module. Users can use it directly.			
		- output/ **other files that generated from mapping files and MIMIC dataset (we attach these files here, user could use our provided scripts to generate)**
			- atc3toSMILES.pkl: drug ID (we use ATC-3 level code to represent drug ID) to drug SMILES string dict
			- records_final.pkl: The final diagnosis-procedure-medication EHR records of each patient, used for train/val/test split. Under MIMIC Dataset policy, we are not allowed to distribute the datasets. Practioners could go to https://physionet.org/content/mimiciv/2.2/ and requrest the access to MIMIC-III dataset and then run our processing script to get the complete preprocessed dataset file.
			- ddi_A_final.pkl: ddi adjacency matrix
			- ehr_adj_final: EHR adjacency matrix
			- ddi_mask_H.pkl: H mask structure
			- voc_final.pkl: diag/prod/med index to code vocabulary
			- substructure_smiles.pkl: med substructure to SMILES mapping
		- dataset processing scripts
			- processing.py
			- get_SMILES.py
	- src/		
		- HKGRec.py: train/test HKGRec
		- models.py: full model of HKGRec and baseline models
		- util.py: setting file
		- layer.py: setting file
- KG_module/ 
    - data/
        - n-ary-1.11.json: the hyper-relational facts.
        - data.ipynb: convert hyper-relational facts into learnable n-ary form.
        - processing.pkl: divided into test set, training set, validation set.
    - other files
        - batching.py: replace the tail entity to get negative facts.
        - builddata.py: build facts data.
        - model.py: full model of knowledge_driven encoder.
        - KG.py: knowledge_driven encoder.
        - entity_id.txt: entities in the constructed HKG to ID mapping file.
        - relation_id.txt: relations in the constructed HKG to ID mapping file.
		

### Step 1: Data Extract

- Go to https://physionet.org/content/mimiciv/2.2/ to download the MIMIC-IV_2.2 dataset (You may need to get the certificate)
- go into the folder and unzip three main files (PROCEDURES_ICD.csv.gz, PRESCRIPTIONS.csv.gz, DIAGNOSES_ICD.csv.gz)
- pip install jupyter notebook
	- run patient_hyp.ipynb, get the EHR of patients with hypertension step by step.
	- run med_hyp.ipynb, get the antihypertensive medications used by hypertension patients step by step.

### Step 2: Data Processing

- change the path in processing.py and processing the data to get a complete records_final.pkl

  ```python
  vim processing.py
  
  # line 364-366
  # med_file = './input/PRESCRIPTIONS.csv'
  # diag_file = './input/DIAGNOSES_ICD.csv'
  # procedure_file = './input/PROCEDURES_ICD.csv'
  
  python processing.py
  ```

- run ddi_mask_H.py to get the ddi_mask_H.pkl

  ```python
  python ddi_mask_H.py
  ```

### Step 3: run the code, we use 

```python
python HKRec.py
```
here is the argument:

    usage: HKRec.py [-h] [--Test] [--model_name MODEL_NAME]
                   [--resume_path RESUME_PATH] [--lr LR]
                   [--target_ddi TARGET_DDI] [--kp KP] [--dim DIM]
    
    optional arguments:
      -h, --help            show this help message and exit
      --Test                test mode
      --model_name MODEL_NAME
                            model name
      --resume_path RESUME_PATH
                            resume path
      --lr LR               learning rate
      --batch_size          batch size 
      --emb_dim             dimension size of embedding
      --max_len             max number of recommended medications
      --beam_size           number of ways in beam search

If you cannot run the code on GPU, just change line 117, "cuda" to "cpu".

Please feel free to contact me <2021212134@nwnu.edu.cn> for any question.
