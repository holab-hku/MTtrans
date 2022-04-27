import os
import numpy as np
import pandas as pd
import PATH
import utils


pj = lambda x: os.path.join(utils.data_dir,x)


RP_data_path = {}
RP_data_path['293T'] = pj('df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt')
RP_data_path['muscle'] = pj('df_counts_and_len.TE_sorted.Muscle.with_annot.txt')
RP_data_path['PC3'] = pj('df_counts_and_len.TE_sorted.pc3.with_annot.txt')



for cell_line, csv_path in RP_data_path.items():

    assert os.path.exists(csv_path), f"The raw data file for {cell_line} does not exist!"
    
    print(f"processing {cell_line}...\n")
    UTR_csv = pd.read_csv(os.path.join(utils.script_dir,"util",f"RP_{cell_line}_last100bp_5UTR.csv"),index_col=0)
    RP_raw_data = pd.read_table(csv_path, sep=' ')
    RP_raw_data.loc[:,'T_id'] = RP_raw_data.index.values
    RP_raw_data['log_te'] = np.log(RP_raw_data.te.values)
    RP_raw_data.sort_values('rpkm_rnaseq', ascending=False, inplace=True)

    RP_raw_data = RP_raw_data.merge(UTR_csv,left_on=['T_id'],right_on=['T_id'],suffixes=["",""])
    RP_raw_data.loc[:,'utr_len'] = RP_raw_data.utr.apply(len)
    RP_raw_dedup = RP_raw_data.drop_duplicates(['utr'], keep='first')
    RP_raw_dedup.to_csv(pj(f"RP_{cell_line}_MTL_transfer.csv"))

print("The preprocssing for RP tasks is Finished !!")
print(f"The files are saved to {utils.data_dir}")