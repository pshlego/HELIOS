export Root_Data_Path=/mnt/sdd
export Root_Path=/root/HELIOS
export CUDA_VISIBLE_DEVICES=0
export Index_Location=${Root_Data_Path}/OTT-QAMountSpace
export Embedding_Data_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/passage
export Modelcheckpoint_PATH=${Root_Data_Path}/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/table_segment_to_passage_v2
export PYTHONPATH=${Root_Path}/Algorithms/ColBERT:${Root_Path}/Algorithms

python ${Root_Path}/Algorithms/Ours/models/passage_retriever.py \
--index_name passage_index \
--ids_path ${Embedding_Data_PATH}/index_to_chunk_id.json \
--collection_path ${Embedding_Data_PATH}/collection.tsv \
--checkpoint_path ${Modelcheckpoint_PATH} \
--index_root_path ${Index_Location}/experiments/default/indexes