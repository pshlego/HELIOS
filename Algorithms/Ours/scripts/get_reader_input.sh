export Root_Data_Path=/mnt/sdd
export Root_Path=/root/HELIOS
export Table_Dataset_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json
export Passage_Data_Path=${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json
export Result_PATH=final_graph.jsonl
export Qa_Dataset_PATH=${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json
export Reader_Input_Data_PATH=reader_input_path.json

python ${Root_Path}/Evaluation/get_reader_input.py \
--table_data_path ${Table_Dataset_PATH} \
--passage_data_path ${Passage_Data_Path} \
--results_path ${Result_PATH} \
--qa_data_path ${Qa_Dataset_PATH} \
--reader_input_path ${Reader_Input_Data_PATH}