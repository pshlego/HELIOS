export Root_Data_Path=/mnt/sdd
export Root_Path=/root/HELIOS
export CUDA_VISIBLE_DEVICES=0,1,2,3
export Modelcheckpoint_PATH=${Root_Data_Path}/OTT-QAMountSpace/ModelCheckpoints/Ours/CrossEncoder/baai_edge_reranker
export PYTHONPATH=${Root_Path}/Algorithms/ColBERT:${Root_Path}/Algorithms

python ${Root_Path}/Algorithms/Ours/models/edge_reranker.py \
--checkpoint_path ${Modelcheckpoint_PATH} \
--process_num 4 \
--cutoff_layer 28