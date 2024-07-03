# frailty_nlp_ner

Jupyter based notebooks (must clean up all reference to EHR data) prior to uploading

Workflow below illustrates the processing of electronic health record (EHR) data using Jupyter notebooks integrating open-sourced deep learning libraries including BERT, FinBERT and HuggingFace. 
EHR data is protected, de-identified and sensitive, please consult authors and/or Central Finland Wellbeing for information to request access. 

<img width="482" alt="image" src="https://github.com/jakelin212/frailty_nlp_ner/assets/6772661/3eb25871-887e-47a4-9209-237c57072a9c">

Figure 1 EHR relevant records are encrypted and scrubbed of all personal information. Working with domain experts, we applied keywords to identified relevant EHR entries and performed appropriate subsampling for labelling within Azure Machine Learning Studio. FinBERT library, a pretrained BERT model adopted for the Finnish language, and the Hugging Face associated  transformer libraries are integrated for the processing, training and evaluation of the labelled and approved EHR text data.. Processing scripts are accomplished with Jupyter noteboods while inference and statistical analysis are conducted  in R .![image](https://github.com/jakelin212/frailty_nlp_ner/assets/6772661/5643d7ab-81a5-4b68-a469-036c04733d79)


