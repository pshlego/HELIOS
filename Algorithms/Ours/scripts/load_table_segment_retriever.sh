export Root_Data_Path=/mnt/sdd
export Root_Path=/root/HELIOS
export CUDA_VISIBLE_DEVICES=3
export Index_Location=${Root_Data_Path}/OTT-QAMountSpace
export Embedding_Data_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/table
export Modelcheckpoint_PATH=${Root_Data_Path}/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/passage_to_table_segment_v2
export PYTHONPATH=${Root_Path}/Algorithms/ColBERT:${Root_Path}/Algorithms

python ${Root_Path}/Algorithms/Ours/models/table_segment_retriever.py \
--index_name table_segment_index \
--ids_path ${Embedding_Data_PATH}/index_to_chunk_id.json \
--collection_path ${Embedding_Data_PATH}/collection.tsv \
--index_root_path ${Index_Location}/experiments/default/indexes \
--checkpoint_path ${Modelcheckpoint_PATH}