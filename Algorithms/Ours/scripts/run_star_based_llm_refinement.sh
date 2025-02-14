export Root_Data_Path=/mnt/sdd
export Root_Path=/root/HELIOS
export Qa_Dataset_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json
export Table_Dataset_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json
export Passage_Data_Path=${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json
export Retrieval_Results_Path=expanded_bipartite_subgraph.jsonl
export Graph_Data_Path=${Root_Data_Path}/OTT-QAMountSpace/Dataset/Ours/Early_Fused_Results/table_chunks_to_passages_cos_table_passage.jsonl
export Result_PATH=final_graph.jsonl

python ${Root_Path}/Algorithms/Ours/star_based_llm_refinement.py \
qa_dataset_path=${Qa_Dataset_PATH} \
table_data_path=${Table_Dataset_PATH} \
passage_data_path=${Passage_Data_Path} \
retrieval_results_path=${Retrieval_Results_Path} \
graph_data_path=${Graph_Data_Path} \
final_result_path=${Result_PATH}