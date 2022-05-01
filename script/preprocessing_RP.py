import os
import numpy as np
import pandas as pd
import PATH
import utils
from Bio import SeqIO
from scipy import sparse


pj = lambda x: os.path.join(utils.data_dir,x)


# read in the sequence 
fa = SeqIO.parse(pj('gencode_v17_5utr_15bpcds.fa'),'fasta')
tx_seq=dict()
for seq_record in fa:
    tx=seq_record.id
    seq=seq_record.seq
    if len(seq)<30: #skip when it too short
        continue
    if "ATG" not in seq:
        continue
    if tx not in tx_seq or len(seq)>len(tx_seq[tx]):
        tx_seq[tx]=seq

# txt records TE and rkpm
RP_data_path = {}
RP_data_path['muscle'] = pj('df_counts_and_len.TE_sorted.Muscle.with_annot.txt')
RP_data_path['PC3'] = pj('df_counts_and_len.TE_sorted.pc3.with_annot.txt')
RP_data_path['293T'] = pj('df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt')

# read the hand crafted features
sparse_feature = os.path.join(utils.script_dir,'util/input_sparseFeature.txt')
test_read = pd.read_table(sparse_feature,header=None,names=['row','column','data'])
row = test_read.row.values
column = test_read.column.values
data = test_read.data.values

# mannual specify the csr matrix
csr_feat_mat = sparse.csr_matrix((data, (row, column)), shape=(57415, 5493)) # index + 1

with open(os.path.join(utils.script_dir, 'util/input.fa.sparseFeature.colname') ,'r') as f:
    columns = [line.split('\t')[1].strip() for line in f]
    f.close

with open(os.path.join(utils.script_dir, 'util/input.fa.sparseFeature.rowname') ,'r') as f:
    rows = [line.split('\t')[1].strip() for line in f]
    f.close

# conver the sparse matrix to dense dataframe
feat_mat = pd.DataFrame(data = csr_feat_mat.todense(), index=rows, columns = columns)
feat_mat['seq'] = [tx_seq[x].upper().__str__() for x in feat_mat.index.values]
feat_mat = feat_mat[['seq']+list(feat_mat.columns)[:32]]

processed_dict = {}
for cell_line, csv_path in RP_data_path.items():

    assert os.path.exists(csv_path), f"The raw data file for {cell_line} does not exist!"
    
    print(f"processing {cell_line}...\n")
    # kmer_data = pd.read_csv(os.path.join(utils.script_dir,"util",f"RP_{cell_line}_kmer_feature.csv"),index_col=0)
    RP_raw_data = pd.read_table(csv_path, sep=' ')
    RP_raw_data.loc[:,'T_id'] = RP_raw_data.index.values
    # adding sequence in 
    RP_raw_data = RP_raw_data.query('rpkm_rnaseq >5 & rpkm_riboseq > 0.1')
    RP_raw_data['log_te'] = np.log(RP_raw_data.te.values)

    RP_feat_merge = RP_raw_data.merge(feat_mat,left_on=['T_id'],right_index=True,suffixes=["",""])
    RP_feat_merge.sort_values('rpkm_rnaseq', ascending=False, inplace=True)
    
    # RP_feat_merge.loc[:,'utr_len'] = RP_feat_merge.utr.apply(len)
    RP_raw_dedup = RP_feat_merge.drop_duplicates(RP_feat_merge.columns[17:], keep='first')
    RP_raw_dedup['utr'] = RP_raw_dedup['seq'].apply(lambda x: x[-216:-16]) # max len 200
    RP_raw_dedup['utr_len'] = RP_raw_dedup.utr.apply(len)
    RP_raw_dedup.query('`utr_len`>30')
    RP_raw_dedup = RP_raw_dedup.drop_duplicates(['utr'], keep='first') 
    # RP_raw_dedup.to_csv(pj(f"RP_{cell_line}_MTL_transfer.csv"))
    processed_dict[cell_line] = RP_raw_dedup

test_set_tid = []
trainval_dict = {}
test_set_dict = {}
for cell_line in ['muscle',"PC3","293T"]:

    seed = 41

    df = processed_dict[cell_line]
    df_to_sample = df.query("`T_id` not in @test_set_tid") 
    df_overlap = df.query("(`T_id` in @test_set_tid)")

    # the number to sample reduce as the list cumulated from previous cell type
    # what to sample 
    n_2_sample = int(0.1*df.shape[0]) - df_overlap.shape[0]
    sampled_subset = df_to_sample.sample(n=n_2_sample,random_state=seed)

    # merge newly sampled with those in the list
    test_set_df = sampled_subset.append(df_overlap)
    trainval_dict[cell_line] = pd.concat([df,test_set_df]).drop_duplicates(keep=False)
    test_set_dict[cell_line] = test_set_df
    # adding new Tids
    test_set_tid += sampled_subset.T_id.values.tolist() 

for cell_line in ["muscle","PC3","293T"]:
    trainval_dict[cell_line].to_csv(pj(f"RP_{cell_line}_train_val.csv"))
    test_set_dict[cell_line].to_csv(pj(f"RP_{cell_line}_test.csv"))

print("The preprocssing for RP tasks is Finished !!")
print(f"The files are saved to {utils.data_dir}")