export Root_Data_Path=/mnt/sdd
export Root_Path=/root/HELIOS
export CUDA_VISIBLE_DEVICES=0,1,2,3
export Index_Location=${Root_Data_Path}/OTT-QAMountSpace
export Embedding_Data_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/edge
export Edge_Dataset_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/Ours/Early_Fused_Results/preprocess_early_fused_results.jsonl
export Modelcheckpoint_PATH=${Root_Data_Path}/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/edge_v3
export PYTHONPATH=${Root_Path}/Algorithms/ColBERT:${Root_Path}/Algorithms

python ${Root_Path}/Algorithms/Ours/models/edge_retriever.py \
--index_name edge_index \
--ids_path ${Embedding_Data_PATH}/index_to_chunk_id.json \
--collection_path ${Embedding_Data_PATH}/collection.tsv \
--index_root_path ${Index_Location}/experiments/default/indexes \
--checkpoint_path ${Modelcheckpoint_PATH} \
--edge_dataset_path ${Edge_Dataset_PATH}