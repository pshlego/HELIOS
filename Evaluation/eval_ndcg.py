import re
import json
import math
import copy
import argparse
from tqdm import tqdm

def compute_dcg(edge_list, gold_edge, k):
    dcg = 0.0
    for i in range(min(k, len(edge_list))):
        rel = 1 if edge_list[i] in gold_edge else 0
        dcg += rel / (math.log2(i + 2))
    return dcg

def compute_idcg(gold_edge, k):
    idcg = 0.0
    for i in range(min(k, len(gold_edge))):
        idcg += 1 / (math.log2(i + 2))
    return idcg
        
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def evaluate(retrieved_graph, qa_data, passage_key_to_content, table_chunk_id_to_table_key, star_graph_dict, edge_limit):
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


    gold_edge_id_list = []
    for positive_ctx in qa_data['positive_ctxs']:
       chunk_id = positive_ctx['chunk_id']
       table_key = table_chunk_id_to_table_key[chunk_id]
       answer_node_list = positive_ctx['answer_node']
       for answer_node in answer_node_list:
            real_row_id = answer_node[1][0]
            row_id = positive_ctx['rows'].index(real_row_id)
            if answer_node[3] == 'passage':
                passage_title = answer_node[0]
                if passage_title not in passage_key_to_content: continue
                gold_edge_id = f"{table_key}_{row_id}_{passage_title}"
                gold_edge_id_list.append(gold_edge_id)
            else:
                try:
                    star_graph = star_graph_dict[f'{table_key}_{row_id}']
                except:
                    continue
                
                for key, value in star_graph['mentions_in_row_info_dict'].items():
                    if value['mention_linked_entity_id_list'] == []: continue
                    passage_title = value['mention_linked_entity_id_list'][0]
                    if passage_title not in passage_key_to_content: continue
                    gold_edge_id = f"{table_key}_{row_id}_{passage_title}"
                    gold_edge_id_list.append(gold_edge_id)

    if gold_edge_id_list == []: return 0, 0
    
    sorted_retrieved_graph = sorted(revised_retrieved_graph.items(), key = lambda x: x[1]['score'], reverse = True)
    edge_id_list = []
    edge_id_set = set()
    for node_id, node_info in sorted_retrieved_graph:
        if node_info['type'] == 'table segment':
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
            passage_content = passage_key_to_content[max_linked_node_id]
            passage_title = passage_content['title']
            edge_id = f"{node_id}_{passage_title}"
            if edge_id in edge_id_set: continue
            edge_id_set.add(edge_id)
            edge_id_list.append(edge_id)

        elif node_info['type'] == 'passage':
            linked_node_id_to_score = {}
            for linked_node_info in node_info['linked_nodes']:
                if linked_node_info[0] not in linked_node_id_to_score:
                    linked_node_id_to_score[linked_node_info[0]] = linked_node_info[1]
                else:
                    if linked_node_info[2] != 'llm_selected':
                        if linked_node_info[1] > linked_node_id_to_score[linked_node_info[0]]:
                            linked_node_id_to_score[linked_node_info[0]] = linked_node_info[1]
                    else:
                        linked_node_id_to_score[linked_node_info[0]] += linked_node_info[1]

            max_linked_node_id = max(linked_node_id_to_score, key=linked_node_id_to_score.get)
            passage_content = passage_key_to_content[node_id]
            passage_title = passage_content['title']
            edge_id = f"{max_linked_node_id}_{passage_title}"
            if edge_id in edge_id_set: continue
            edge_id_set.add(edge_id)
            edge_id_list.append(edge_id)

    dcg_k = compute_dcg(edge_id_list, gold_edge_id_list, edge_limit)
    idcg_k = compute_idcg(gold_edge_id_list, edge_limit)
    ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0
    
    return ndcg_k, idcg_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Flask app with custom index and ids paths")
    parser.add_argument('--table_data_path', type=str, required=True, help='Table data path')
    parser.add_argument('--passage_data_path', type=str, required=True, help='Passage data path')
    parser.add_argument('--ground_truth_data_graph_path', type=str, required=True, help='Ground truth data graph path')
    parser.add_argument('--results_path', type=str, required=True, help='Results path')
    parser.add_argument('--top_k', type=int, required=True, help='Top k')
    args = parser.parse_args()

    ground_truth_data_graph_path = args.ground_truth_data_graph_path
    results_path = args.results_path
    table_data_path = args.table_data_path
    passage_data_path = args.passage_data_path
    top_k = args.top_k

    print(results_path)
    print("1. Loading tables...")
    table_key_to_content = {}
    table_chunk_id_to_table_key = {}
    table_contents = json.load(open(table_data_path))
    print("1. Loaded " + str(len(table_contents)) + " tables!")
    print("2. Processing tables...")
    for table_key, table_content in tqdm(enumerate(table_contents)):
        table_key_to_content[str(table_key)] = table_content
        table_chunk_id_to_table_key[table_content['chunk_id']] = str(table_key)
    print("2. Processing tables complete!", end = "\n\n")
    print("3. Loading passages...")
    passage_key_to_content = {}
    passage_contents = json.load(open(passage_data_path))
    print("3. Loaded " + str(len(passage_contents)) + " passages!")
    print("4. Processing passages...")
    for passage_content in tqdm(passage_contents):
        passage_key_to_content[passage_content['title']] = passage_content
    print("4. Processing passages complete!", end = "\n\n")
    print("5. Loading ground truth star graphs...")
    gt_star_graph_list = read_jsonl(ground_truth_data_graph_path)
    gt_star_graph_dict = {gt_star_graph['chunk_id']: gt_star_graph for gt_star_graph in gt_star_graph_list}
    print("5. Loaded " + str(len(gt_star_graph_list)) + " ground truth star graphs!")
    print("6. Loading retrieved results...")
    retrieved_results = read_jsonl(results_path)
    print("6. Loaded " + str(len(retrieved_results)) + " retrieved results!")

    ndcg_list = []
    for retrieved_result in tqdm(retrieved_results):
        qa_data = retrieved_result["qa data"]
        retrieved_graph = retrieved_result["retrieved graph"]
        ndcg_k, idcg_k  = evaluate(retrieved_graph, qa_data, passage_key_to_content, table_chunk_id_to_table_key, gt_star_graph_dict, top_k)
        if ndcg_k == 0 and idcg_k == 0: continue
        ndcg_list.append(ndcg_k)
                
    print(f"NDCG@{top_k}: ", sum(ndcg_list)/len(ndcg_list))