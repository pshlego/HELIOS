export Root_Data_Path=/mnt/sdd
export Root_Path=/root/HELIOS
export Qa_Dataset_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json
export Table_Dataset_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json
export Passage_Data_Path=${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json
export Result_PATH=retrieved_bipartite_subgraph.jsonl
export PYTHONPATH=${Root_Path}/Algorithms/ColBERT:${Root_Path}/Algorithms

python ${Root_Path}/Algorithms/Ours/edge_based_bipartite_subgraph_retrieval.py \
qa_dataset_path=${Qa_Dataset_PATH} \
table_data_path=${Table_Dataset_PATH} \
passage_data_path=${Passage_Data_Path} \
final_result_path=${Result_PATH}
