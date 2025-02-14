import re
import json
import copy
import unicodedata
from tqdm import tqdm
import argparse
from dpr.data.qa_validation import has_answer
from dpr.utils.tokenizers import SimpleTokenizer
        
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def evaluate(retrieved_graph, qa_data, table_key_to_content, passage_key_to_content, tokenizer):
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


    node_count = 0
    edge_count = 0
    answers = qa_data['answers']
    context = ""
    retrieved_table_set = set()
    retrieved_passage_set = set()
    sorted_retrieved_graph = sorted(revised_retrieved_graph.items(), key = lambda x: x[1]['score'], reverse = True)
    
    for node_id, node_info in sorted_retrieved_graph:
        if node_info['type'] == 'table segment':
            
            table_id = node_id.split('_')[0]
            table = table_key_to_content[table_id]
            chunk_id = table['chunk_id']
            node_info['chunk_id'] = chunk_id
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                
                context += table['text']
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
            
            edge_count += 1
            context += edge_text
        
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
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                context += table['text']
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
            context += edge_text
            edge_count += 1

    normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
    normalized_context = remove_accents_and_non_ascii(context)
    is_has_answer = has_answer(normalized_answers, normalized_context, tokenizer, 'string', max_length=4096)
    
    if is_has_answer:
        recall = 1
        error_analysis = copy.deepcopy(qa_data)
        error_analysis['retrieved_graph'] = retrieved_graph
        error_analysis['sorted_retrieved_graph'] = sorted_retrieved_graph[:node_count]
        if  "hard_negative_ctxs" in error_analysis:
            del error_analysis["hard_negative_ctxs"]
    else:
        recall = 0
        error_analysis = copy.deepcopy(qa_data)
        error_analysis['retrieved_graph'] = retrieved_graph
        error_analysis['sorted_retrieved_graph'] = sorted_retrieved_graph[:node_count]
        if  "hard_negative_ctxs" in error_analysis:
            del error_analysis["hard_negative_ctxs"]

    return recall, error_analysis

def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Flask app with custom index and ids paths")
    parser.add_argument('--table_data_path', type=str, required=True, help='Table data path')
    parser.add_argument('--passage_data_path', type=str, required=True, help='Passage data path')
    parser.add_argument('--results_path', type=str, required=True, help='Results path')
    args = parser.parse_args()
    table_data_path = args.table_data_path
    passage_data_path = args.passage_data_path
    results_path = args.results_path
    print("3. Loading tables...")
    table_key_to_content = {}
    table_contents = json.load(open(table_data_path))
    print("3. Loaded " + str(len(table_contents)) + " tables!")
    print("3. Processing tables...")
    for table_key, table_content in tqdm(enumerate(table_contents)):
        table_key_to_content[str(table_key)] = table_content
    print("3. Processing tables complete!", end = "\n\n")
    print("4. Loading passages...")
    passage_key_to_content = {}
    passage_contents = json.load(open(passage_data_path))
    print("4. Loaded " + str(len(passage_contents)) + " passages!")
    print("4. Processing passages...")
    for passage_content in tqdm(passage_contents):
        passage_key_to_content[passage_content['title']] = passage_content
    print("4. Processing passages complete!", end = "\n\n")
    retrieved_results = read_jsonl(results_path)
    recall_list = []
    tokenizer = SimpleTokenizer()
    for retrieved_result in tqdm(retrieved_results):
        qa_data = retrieved_result["qa data"]
        retrieved_graph = retrieved_result["retrieved graph"]
        recall, error_case  = evaluate(retrieved_graph, qa_data, table_key_to_content, passage_key_to_content, tokenizer)
        recall_list.append(recall)
            
    print("HIT@4K: ", sum(recall_list)/len(recall_list))