DATA_DIR=$(cat machine_configure.json |grep data_dir|awk '{print $2}'|awk -F "," '{print $1}')

if [! -d ${DATA_DIR} ]; then
    DATA_DIR=$(pwd)/data

cd $DATA_DIR
# download the ribosome profiling datasest 
# RP-muscle
wget https://github.com/zzz2010/5UTR_Optimizer/blob/master/data/df_counts_and_len.TE_sorted.Muscle.with_annot.txt &
# RP-293T
wget https://github.com/zzz2010/5UTR_Optimizer/blob/master/data/df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt &
# RP-PC3
wget https://github.com/zzz2010/5UTR_Optimizer/blob/master/data/df_counts_and_len.TE_sorted.pc3.with_annot.txt &


# download the Massively parallel report assay datasets
# MPA_U
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130435/suppl/GSM3130435_egfp_unmod_1.csv.gz 
# MPA_H
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130443/suppl/GSM3130443_designed_library.csv.gz 
# MPA_V
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4084nnn/GSM4084997/suppl/GSM4084997_varying_length_25to100.csv.gz


# dowlaod the yeast dataset
wget ftp.ncbi.nlm.nih.gov/geo/samples/GSM2793nnn/GSM2793752/suppl/GSM2793752_Random_UTRs.csv.gz


gzip -d *.gz
echo "Finished ! All data ready~"
echo "Please step to the pre-proccessing"


