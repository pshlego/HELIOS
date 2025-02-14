export Root_Data_Path=/mnt/sdd
export Root_Path=/root/HELIOS
export CUDA_VISIBLE_DEVICES=0,1,2,3
export Index_Location=${Root_Data_Path}/OTT-QAMountSpace
export Embedding_Data_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/table
export Modelcheckpoint_PATH=${Root_Data_Path}/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/passage_to_table_segment_v2
export PYTHONPATH=${Root_Path}/Algorithms/ColBERT:${Root_Path}/Algorithms

cd ${Index_Location}
python ${Root_Path}/Algorithms/Ours/build_index.py \
collection_root_dir_path=${Embedding_Data_PATH} \
collection_tsv_path=${Embedding_Data_PATH}/collection.tsv \
colbert_checkpoint=${Modelcheckpoint_PATH} \
query_maxlen=96 \
index_name=table_segment_index