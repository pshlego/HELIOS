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

@hydra.main(config_path = "conf", config_name = "edge_based_bipartite_subgraph_retrieval")
def main(cfg: DictConfig):
    # load qa dataset
    print()
    print(f"[[ Loading qa dataset... ]]", end = "\n\n")
    qa_dataset = json.load(open(cfg.qa_dataset_path))
    print(f"Number of questions: {len(qa_dataset)}", end = "\n\n")

    # load graph query engine
    print(f"[[ Loading edge-based bipartite subgraph retriever... ]]", end = "\n\n")
    bipartite_subgraph_retriever = BipartiteSubgraphRetriever(cfg)
    print(f"Edge-based bipartite subgraph loaded successfully!", end = "\n\n")
    print(f"Start querying...")
    for qidx, qa_datum in tqdm(enumerate(qa_dataset), total = len(qa_dataset)):
        question = qa_datum['question']
        retrieved_graph = bipartite_subgraph_retriever.retrieve(question)

        # save retrieved graph
        to_print = {
            "qa data": qa_datum,
            "retrieved graph": retrieved_graph
        }
        
        with open(cfg.final_result_path, 'a+') as file:
            file.write(json.dumps(to_print) + '\n')

    print(f"Querying done.")

class BipartiteSubgraphRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.edge_retriever_addr = "http://localhost:5000/edge_retrieve"
        self.edge_reranker_addr = "http://localhost:5001/edge_rerank"

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

    def retrieve(self, question):
        # 1.1 Retrieve edges
        retrieved_edges = self.retrieve_edges(question)

        # 1.2 Rerank edges
        reranked_edges = self.rerank_edges(question, retrieved_edges)

        # 1.3 Integrate edges into bipartite subgraph candidates
        bipartite_subgraph_candidates = self.integrate_into_graph(reranked_edges)

        return bipartite_subgraph_candidates

    def retrieve_edges(self, question):
        response = requests.post(
            self.edge_retriever_addr,
            json={
                "query": question,
                "k": self.cfg.top_k_of_retrieved_edges
            },
            timeout=None,
        ).json()

        retrieved_edges = response['edge_content_list']

        return retrieved_edges
    
    def rerank_edges(self, question, retrieved_edges):
        model_input = []
        for retrieved_edge in retrieved_edges:
            edge_text = self.get_edge_text(retrieved_edge)
            model_input.append([question, edge_text])
        
        response = requests.post(
            self.edge_reranker_addr,
            json={
                "model_input": model_input,
                "max_length": 256
            },
            timeout=None,
        ).json()

        model_input = response['model_input']
        reranking_scores = response['reranking_scores']
        for retrieved_edge, reranking_score in tqdm(zip(retrieved_edges, reranking_scores), total = len(retrieved_edges)):
            retrieved_edge['reranking_score'] = float(reranking_score)
        
        # Sort edges by reranking score
        reranked_edges = sorted(retrieved_edges, key = lambda x: x['reranking_score'], reverse = True)[:self.cfg.top_k_of_reranked_edges]

        return reranked_edges

    def integrate_into_graph(self, reranked_edges):
        bipartite_subgraph_candidates = {}
        
        for reranked_edge in reranked_edges:
            if 'linked_entity_id' in reranked_edge:
                # get edge info
                table_key = str(reranked_edge['table_id'])
                row_id = reranked_edge['chunk_id'].split('_')[1]
                table_segment_node_id = f"{table_key}_{row_id}"
                passage_id = reranked_edge['linked_entity_id']
                
                # get edge score
                edge_score = reranked_edge['reranking_score']

                # add nodes
                self.add_node(bipartite_subgraph_candidates, 'table segment', table_segment_node_id, passage_id, edge_score, 'edge_reranking')
                self.add_node(bipartite_subgraph_candidates, 'passage', passage_id, table_segment_node_id, edge_score, 'edge_reranking')
        
        return bipartite_subgraph_candidates

    def get_edge_text(self,  edge):
        table_key = edge['chunk_id'].split('_')[0]
        row_id = int(edge['chunk_id'].split('_')[1])
        table_content = self.table_key_to_content[table_key]
        table_title = table_content['title']
        table_rows = table_content['text'].split('\n')
        column_names = table_rows[0]
        row_values = table_rows[row_id+1]
        table_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + row_values
        passage_key = edge['linked_entity_id']
        passage_text = self.passage_key_to_content[passage_key]['text']
        edge_text = table_text + ' [SEP] ' + passage_text
        return edge_text
    
    def add_node(self, graph, source_node_type, source_node_id, target_node_id, score, retrieval_type, source_rank = 0, target_rank = 0):
        if source_node_id not in graph:
            graph[source_node_id] = {'type': source_node_type, 'linked_nodes': [[target_node_id, score, retrieval_type, source_rank, target_rank]]}
        else:
            graph[source_node_id]['linked_nodes'].append([target_node_id, score, retrieval_type, source_rank, target_rank])

if __name__ == "__main__":
    main()