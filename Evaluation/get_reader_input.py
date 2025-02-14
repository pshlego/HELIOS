import re
import json
import copy
import argparse
import unicodedata
from tqdm import tqdm
from dpr.utils.tokenizers import SimpleTokenizer

def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Flask app with custom index and ids paths")
    parser.add_argument('--table_data_path', type=str, required=True, help='Table data path')
    parser.add_argument('--passage_data_path', type=str, required=True, help='Passage data path')
    parser.add_argument('--results_path', type=str, required=True, help='Results path')
    parser.add_argument('--qa_data_path', type=str, required=True, help='QA data path')
    parser.add_argument('--reader_input_path', type=str, required=True, help='Reader input data path')
    args = parser.parse_args()
    table_data_path = args.table_data_path
    passage_data_path = args.passage_data_path
    results_path = args.results_path
    qa_dataset_path=  args.qa_data_path
    reader_input_path = args.reader_input_path
    print(f"Loading corpus...")
    table_key_to_content = {}
    table_contents = json.load(open(table_data_path))
    for table_key, table_content in enumerate(table_contents):
        table_key_to_content[str(table_key)] = table_content
    
    passage_key_to_content = {}
    passage_contents = json.load(open(passage_data_path))
    for passage_content in passage_contents:
        passage_key_to_content[passage_content['title']] = passage_content

    retrieved_graphs = read_jsonl(results_path)
    qa_dataset = json.load(open(qa_dataset_path))

    reader_input_list = []
    revised_retrieved_graphs = []
    total_recall_dict = {}
    tokenizer = SimpleTokenizer()
    for retrieved_graph_info in retrieved_graphs:
        retrieved_graph = retrieved_graph_info['retrieved graph']
        revised_retrieved_graph = {}
        for node_id, node_info in retrieved_graph.items():                  
            wo_llm_selected = [x for x in node_info['linked_nodes'] if x[2] != 'llm_selected']
            linked_nodes = node_info['linked_nodes']

            if len(wo_llm_selected) == 0: continue
            if len(linked_nodes) == 0: continue
            
            revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
            revised_retrieved_graph[node_id]['linked_nodes'] = linked_nodes
            linked_scores = [linked_node[1] for linked_node in linked_nodes]

            node_score = max(linked_scores)
            
            if node_score == 1000000:
                additional_score_list = [linked_node[1] for linked_node in linked_nodes if linked_node[2] != 'llm_selected']
                if len(additional_score_list) > 0:
                    node_score += max(additional_score_list)

            revised_retrieved_graph[node_id]['score'] = node_score
        revised_retrieved_graphs.append(revised_retrieved_graph)

    for revised_retrieved_graph, qa_datum in tqdm(zip(revised_retrieved_graphs, qa_dataset), total=len(qa_dataset)):
        cos_format_result = copy.deepcopy(qa_datum)
        edge_count = 0
        all_included = []
        retrieved_table_set = set()
        retrieved_passage_set = set()
        sorted_retrieved_graph = sorted(revised_retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
        
        for node_id, node_info in sorted_retrieved_graph:
            if node_info['type'] == 'table segment':
                
                table_id = node_id.split('_')[0]
                table = table_key_to_content[table_id]
                chunk_id = table['chunk_id']
                node_info['chunk_id'] = chunk_id
                
                if table_id not in retrieved_table_set:
                    retrieved_table_set.add(table_id)
                    
                    all_included.append({'id': chunk_id, 'title': table['title'], 'text': table['text']})
                    
                    edge_count += 1
                    
                linked_node_id_to_score = {}
                for linked_node_info in node_info['linked_nodes']:
                    if linked_node_info[0] not in linked_node_id_to_score:
                        linked_node_id_to_score[linked_node_info[0]] = linked_node_info[1]
                    else:
                        if linked_node_info[2] != 'llm_selected':
                            if linked_node_info[1] > linked_node_id_to_score[linked_node_info[0]]:
                                linked_node_id_to_score[linked_node_info[0]] = linked_node_info[1]
                        else:
                            if linked_node_id_to_score[linked_node_info[0]] < 500000:
                                linked_node_id_to_score[linked_node_info[0]] += linked_node_info[1]
                max_linked_node_id = max(linked_node_id_to_score, key=linked_node_id_to_score.get)
                
                if max_linked_node_id in retrieved_passage_set:
                    continue
                
                retrieved_passage_set.add(max_linked_node_id)
                passage_content = passage_key_to_content[max_linked_node_id]
                passage_text = passage_content['title'] + ' ' + passage_content['text']
                
                row_id = int(node_id.split('_')[1])
                table_rows = table['text'].split('\n')
                column_name = table_rows[0]
                row_values = table_rows[row_id+1]
                table_segment_text = column_name + '\n' + row_values
                
                edge_text = table_segment_text + '\n' + passage_text
                
                all_included.append({'id': chunk_id, 'title': table['title'], 'text': edge_text})
                edge_count += 1
                
            elif node_info['type'] == 'passage':
                if node_id in retrieved_passage_set:
                    continue

                linked_node_id_to_score = {}
                for linked_node_info in node_info['linked_nodes']:
                    if linked_node_info[0] not in linked_node_id_to_score:
                        linked_node_id_to_score[linked_node_info[0]] = linked_node_info[1]
                    else:
                        if linked_node_info[2] != 'llm_selected':
                            if linked_node_info[1] > linked_node_id_to_score[linked_node_info[0]]:
                                linked_node_id_to_score[linked_node_info[0]] = linked_node_info[1]
                        else:
                            if linked_node_id_to_score[linked_node_info[0]] < 500000:
                                linked_node_id_to_score[linked_node_info[0]] += linked_node_info[1]
                max_linked_node_id = max(linked_node_id_to_score, key=linked_node_id_to_score.get)
                
                table_id = max_linked_node_id.split('_')[0]
                table = table_key_to_content[table_id]
                chunk_id = table['chunk_id']
                
                if table_id not in retrieved_table_set:
                    retrieved_table_set.add(table_id)
                    all_included.append({'id': chunk_id, 'title': table['title'], 'text': table['text']})
                    edge_count += 1

                row_id = int(max_linked_node_id.split('_')[1])
                table_rows = table['text'].split('\n')
                column_name = table_rows[0]
                row_values = table_rows[row_id+1]
                table_segment_text = column_name + '\n' + row_values

                retrieved_passage_set.add(node_id)
                passage_content = passage_key_to_content[node_id]
                passage_text = passage_content['title'] + ' ' + passage_content['text']
                
                edge_text = table_segment_text + '\n' + passage_text
                all_included.append({'id': chunk_id, 'title': table['title'], 'text': edge_text})
                edge_count += 1
        
        cos_format_result['ctxs'] = all_included
        reader_input_list.append(cos_format_result)
    
    with open(reader_input_path, 'w') as f:
        json.dump(reader_input_list, f, indent=4)