import re
import json
import copy
import unicodedata
import ast
import json
import time
import hydra
import torch
import concurrent.futures
import requests
from tqdm import tqdm
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import set_seed
from utils.utils import read_jsonl
set_seed(0)

@hydra.main(config_path = "conf", config_name = "query_relevant_node_expansion")
def main(cfg: DictConfig):
    # load qa dataset
    print()
    print(f"[[ Loading qa dataset... ]]", end = "\n\n")
    qa_dataset = json.load(open(cfg.qa_dataset_path))
    print(f"Number of questions: {len(qa_dataset)}", end = "\n\n")

    # load graph query engine
    print(f"[[ Loading query-relevant node expansion... ]]", end = "\n\n")
    node_expander = NodeExpander(cfg)
    print(f"Query-relevant node expansion loaded successfully!", end = "\n\n")
    retrieved_results = read_jsonl(cfg.retrieval_results_path)

    print(f"Start querying...")
    for qidx, (qa_datum, retrieved_result) in tqdm(enumerate(zip(qa_dataset, retrieved_results)), total = len(qa_dataset)):
        question = qa_datum['question']
        retrieved_graph = retrieved_result['retrieved graph']
        expanded_graph = node_expander.expand(question, retrieved_graph)

        # save retrieved graph
        to_print = {
            "qa data": qa_datum,
            "retrieved graph": expanded_graph
        }
        
        with open(cfg.final_result_path, 'a+') as file:
            file.write(json.dumps(to_print) + '\n')

    print(f"Querying done.")

class NodeExpander:
    def __init__(self, cfg):
        self.cfg = cfg
        self.edge_reranker_addr = "http://localhost:5001/edge_rerank"
        self.node_scorer_addr = "http://localhost:5002/node_score"
        self.table_segment_retriever_addr_list = ["http://localhost:5003/table_segment_retrieve"]
        self.passage_retriever_addr_list = ["http://localhost:5004/passage_retrieve"]

        # 1. Load tables
        print("1. Loading tables...")
        table_contents = json.load(open(cfg.table_data_path))
        TABLES_NUM = len(table_contents)
        self.table_key_to_content = {str(table_key): table_content for table_key, table_content in tqdm(enumerate(table_contents), total = TABLES_NUM)}
        print("1. Loaded " + str(TABLES_NUM) + " tables!", end = "\n\n")

        # 2. Load passages
        print("2. Loading passages...")
        passage_contents = json.load(open(cfg.passage_data_path))
        PASSAGES_NUM = len(passage_contents)
        self.passage_key_to_content = {str(passage_content['title']): passage_content for passage_content in tqdm(passage_contents)}
        print("2. Loaded " + str(PASSAGES_NUM) + " passages!", end = "\n\n")

        self.process_num = len(self.table_segment_retriever_addr_list)

    def expand(self, question, retrieved_graph):
        # 2.1 Seed Node Selection
        retrieved_graph, topk_selected_nodes = self.select_seed_node(question, retrieved_graph)
        
        # 2.2 Expanding Node Selection
        self.select_expanding_node(question, retrieved_graph, topk_selected_nodes)
        
        return retrieved_graph
    
    def select_seed_node(self, question, retrieved_graph):
        node_text_list = []
        node_id_list = []
        for node_id, node_info in retrieved_graph.items():
            if node_info['type'] == 'table segment':
                table_id = node_id.split('_')[0]
                row_id = int(node_id.split('_')[1])
                table_content = self.table_key_to_content[table_id]
                table_title = table_content['title']
                table_rows = table_content['text'].split('\n')
                column_names = table_rows[0]
                row_values = table_rows[row_id+1]
                table_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + row_values
                node_text_list.append(table_text)
                node_id_list.append(node_id)
            elif node_info['type'] == 'passage':
                passage_content = self.passage_key_to_content[node_id]
                passage_text = passage_content['title'] + ' [SEP] ' + passage_content['text']
                node_text_list.append(passage_text)
                node_id_list.append(node_id)
        
        model_input = [[question, node_text] for node_text in node_text_list]
        
        response = requests.post(
            self.node_scorer_addr,
            json={
                "model_input": model_input,
                "max_length": 128
            },
            timeout=None,
        ).json()

        model_input = response['model_input']
        reranking_scores = response['reranking_scores']

        # Assign the computed score to the node
        for node_id, node_score in zip(node_id_list, reranking_scores):
            retrieved_graph[node_id]['score'] = node_score

        node_list = []
        for node_id, node_info in retrieved_graph.items():
            node_list.append((node_id, node_info['score'], node_info['type']))

        topk_node_list = sorted(node_list, key=lambda x: x[1], reverse=True)[:self.cfg.beam_size]

        topk_node_scores = torch.tensor([node[1] for node in topk_node_list])

        topk_node_probs = F.softmax(topk_node_scores, dim=0)

        topk_selected_nodes = [(node[0], prob, node[2]) for node, prob in zip(topk_node_list, topk_node_probs)]

        return retrieved_graph, topk_selected_nodes


    def select_expanding_node(self, nl_question, graph, topk_selected_nodes):
        final_prob_list = []
        expanded_query_list = []
        for (query_node_id, query_node_prob, query_node_type) in topk_selected_nodes:
            expanded_query = self.get_expanded_query(nl_question, query_node_id, query_node_type)
            expanded_query_list.append([expanded_query, query_node_id, query_node_prob, query_node_type])
        
        devided_expanded_query_list = []
        for rank in range(self.process_num):
            devided_expanded_query_list.append(expanded_query_list[rank*len(expanded_query_list)//self.process_num:(rank+1)*len(expanded_query_list)//self.process_num])
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_list = []
            final_prob_list = []
            for rank in range(self.process_num):
                future = executor.submit(self.expanded_query_worker, (self.passage_retriever_addr_list[rank], self.table_segment_retriever_addr_list[rank], devided_expanded_query_list[rank]))
                future_list.append(future)

            concurrent.futures.wait(future_list)
            for future in future_list:
                final_prob_list.extend(future.result())

        final_prob_list = sorted(final_prob_list, key=lambda x: x[2], reverse=True)[:self.cfg.beam_size]

        edge_total_list = []
        for query_node_id, target_node_id, final_prob, query_node_type, target_node_type in final_prob_list:
            if query_node_type == 'table segment':
                table_key = query_node_id.split('_')[0]
                row_id = int(query_node_id.split('_')[1])
                table = self.table_key_to_content[table_key]
                table_title = table['title']
                table_column_names = table['text'].split('\n')[0]
                table_row_values = table['text'].split('\n')[row_id+1]
                query_node_text = f"{table_title} [SEP] {table_column_names} [SEP] {table_row_values}"
                passage_text = self.passage_key_to_content[target_node_id]['text']
                target_text = passage_text
            else:
                passage = self.passage_key_to_content[query_node_id]
                query_node_text = f"{passage['title']} [SEP] {passage['text']}"
                table_key = target_node_id.split('_')[0]
                row_id = int(target_node_id.split('_')[1])
                table_content = self.table_key_to_content[table_key]
                table_title = table_content['title']
                table_rows = table_content['text'].split('\n')
                column_names = table_rows[0]
                row_values = table_rows[1:][row_id]
                target_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + row_values

            edge_text = f"{query_node_text} [SEP] {target_text}"
            edge_total_list.append(edge_text)

        model_input = [[nl_question, edge] for edge in edge_total_list]
        
        response = requests.post(
            self.edge_reranker_addr,
            json={
                "model_input": model_input,
                "max_length": 256
            },
            timeout=None,
        ).json()

        reranking_scores = response['reranking_scores']
        
        for i, (query_node_id, target_node_id, final_prob, query_node_type, target_node_type) in enumerate(final_prob_list):
            reranked_prob = float(torch.tensor(reranking_scores[i]))
            query_rank = [node[0] for node in final_prob_list].index(query_node_id)
            self.add_node(graph, query_node_type, query_node_id, target_node_id, reranked_prob, 'node_augmentation', query_rank, i)
            self.add_node(graph, target_node_type, target_node_id, query_node_id, reranked_prob, 'node_augmentation', i, query_rank)

    def expanded_query_worker(self, worker_input):
        passage_retriever_addr = worker_input[0]
        table_segment_retriever_addr = worker_input[1]
        expanded_query_list = worker_input[2]
        final_prob_list = []
        for expanded_query, query_node_id, query_node_prob, query_node_type in expanded_query_list:
            if query_node_type == 'table segment':
                target_node_type = 'passage'
                response = requests.post(
                    passage_retriever_addr,
                    json={
                        "query": expanded_query,
                        "k": self.cfg.beam_size
                    },
                    timeout=None,
                ).json()
                
                retrieved_node_id_list = response['retrieved_key_list']
                retrieved_score_list = response['retrieved_score_list']
            else:
                target_node_type = 'table segment'
                response = requests.post(
                    table_segment_retriever_addr,
                    json={
                        "query": expanded_query,
                        "k": self.cfg.beam_size
                    },
                    timeout=None,
                ).json()
                
                retrieved_node_id_list = response['retrieved_key_list']
                retrieved_score_list = response['retrieved_score_list']

            retrieved_scores = torch.tensor(retrieved_score_list)
            retrieved_probs = F.softmax(retrieved_scores, dim=0)

            for idx, target_node_id in enumerate(retrieved_node_id_list):
                final_prob = query_node_prob * retrieved_probs[idx]
                final_prob_list.append((query_node_id, target_node_id, final_prob, query_node_type, target_node_type))
        
        return final_prob_list

    def get_expanded_query(self, nl_question, node_id, query_node_type):
        if query_node_type == 'table segment':
            table_key = node_id.split('_')[0]
            table = self.table_key_to_content[table_key]
            table_title = table['title']
            
            row_id = int(node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            
            expanded_query = f"{nl_question} [SEP] {table_title} [SEP] {column_name} [SEP] {row_values}"
        else:
            passage = self.passage_key_to_content[node_id]
            passage_title = passage['title']
            passage_text = passage['text']
            
            expanded_query = f"{nl_question} [SEP] {passage_title} [SEP] {passage_text}"
        
        return expanded_query

    def add_node(self, graph, source_node_type, source_node_id, target_node_id, score, retrieval_type, source_rank = 0, target_rank = 0):
        if source_node_id not in graph:
            graph[source_node_id] = {'type': source_node_type, 'linked_nodes': [[target_node_id, score, retrieval_type, source_rank, target_rank]]}
        else:
            graph[source_node_id]['linked_nodes'].append([target_node_id, score, retrieval_type, source_rank, target_rank])

if __name__ == "__main__":
    main()