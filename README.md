# PRICAI2022-71
The technical Appendix is named pricai22Appendix.pdf.

The codes for replicating benchmark datasets, IHDP experiments and Twins experiments, are packed in the file "pricai22_code". 
Please follow "Replication instruction.txt" to replicate our experimental results.

For IHDP experiments:
The 1000 IHDP datasets can be downloaded from https://www.fredjo.com/
Please open MBRL_train.py and modify relevant parameter flags: 'outdir', 'datadir'. Then remain other parameters the same and directly run MBRL_train.py.
After the python process stops, you can open evaluate.py and directly run it. The final results will be saved as results_summary_purt_rmse_fact_0.1.txt.

For Twins experiments:
The Twins dataset can be downloaded from https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS. 
100 Twins dataset is generated using the raw Twins dataset, and the specific data generating process is the same as https://github.com/jsyoon0823/GANITE/blob/master/data_loading.py.
Please open MBRL_train.py and modify relevant parameter flags to remain the parameters the same as in our paper. Then do the same as IHDP experiments to reproduce the results.

