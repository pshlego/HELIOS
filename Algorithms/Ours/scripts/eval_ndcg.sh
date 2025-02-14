export Root_Data_Path=/mnt/sdd
export Root_Path=/root/HELIOS
export Table_Dataset_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json
export Passage_Data_Path=${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json
export GT_Data_Graph_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/Ours/Early_Fused_Results/preprocess_ground_truth_data_graph.jsonl
export Result_PATH=final_graph.jsonl

python ${Root_Path}/Evaluation/eval_ndcg.py \
--table_data_path ${Table_Dataset_PATH} \
--passage_data_path ${Passage_Data_Path} \
--ground_truth_data_graph_path ${GT_Data_Graph_PATH} \
--results_path ${Result_PATH} \
--top_k 50