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
from models.llm_node_selector import LlmNodeSelector
set_seed(0)

@hydra.main(config_path = "conf", config_name = "star_based_llm_refinement")
def main(cfg: DictConfig):
    # load qa dataset
    print()
    print(f"[[ Loading qa dataset... ]]", end = "\n\n")
    qa_dataset = json.load(open(cfg.qa_dataset_path))
    print(f"Number of questions: {len(qa_dataset)}", end = "\n\n")

    # load graph query engine
    print(f"[[ Loading query-relevant node expansion... ]]", end = "\n\n")
    star_refiner = StarRefiner(cfg)
    print(f"Query-relevant node expansion loaded successfully!", end = "\n\n")
    retrieved_results = read_jsonl(cfg.retrieval_results_path)

    print(f"Start querying...")
    for qidx, (qa_datum, retrieved_result) in tqdm(enumerate(zip(qa_dataset, retrieved_results)), total = len(qa_dataset)):
        question = qa_datum['question']
        expanded_graph = retrieved_result['retrieved graph']
        refined_graph = star_refiner.refine(question, expanded_graph)

        # save retrieved graph
        to_print = {
            "qa data": qa_datum,
            "retrieved graph": refined_graph
        }
        
        with open(cfg.final_result_path, 'a+') as file:
            file.write(json.dumps(to_print) + '\n')

    print(f"Querying done.")

class StarRefiner:
    def __init__(self, cfg):
        self.cfg = cfg

        # 1. Load tables
        print("1. Loading tables...")
        table_contents = json.load(open(cfg.table_data_path))
        TABLES_NUM = len(table_contents)
        self.table_key_to_content = {str(table_key): table_content for table_key, table_content in tqdm(enumerate(table_contents), total = TABLES_NUM)}
        self.table_chunk_id_to_table_key = {table_content['chunk_id']: str(table_key) for table_key, table_content in tqdm(enumerate(table_contents), total = TABLES_NUM)}
        print("1. Loaded " + str(TABLES_NUM) + " tables!", end = "\n\n")

        # 2. Load passages
        print("2. Loading passages...")
        passage_contents = json.load(open(cfg.passage_data_path))
        PASSAGES_NUM = len(passage_contents)
        self.passage_key_to_content = {str(passage_content['title']): passage_content for passage_content in tqdm(passage_contents)}
        print("2. Loaded " + str(PASSAGES_NUM) + " passages!", end = "\n\n")

        graph_data = read_jsonl(cfg.graph_data_path)
        self.table_segment_id_to_linked_passage_ids = {}
        for graph_datum in tqdm(graph_data):
            table_chunk_id = graph_datum['table_chunk_id']
            table_key = self.table_chunk_id_to_table_key[table_chunk_id]
            for result in graph_datum['results']:
                row_id = result['row']
                table_segment_id = f"{table_key}_{row_id}"
                if table_segment_id not in self.table_segment_id_to_linked_passage_ids:
                    self.table_segment_id_to_linked_passage_ids[table_segment_id] = []
            
                passage_id = result['retrieved'][0]
                self.table_segment_id_to_linked_passage_ids[table_segment_id].append(passage_id)

        self.llm_node_selector = LlmNodeSelector(cfg, self.table_key_to_content, self.passage_key_to_content)

    def refine(self, question, expanded_graph):
        # decompose expanded graph into star_graphs
        star_graph_list, table_id_to_row_id_to_linked_passage_ids = self.decompose_into_star_graphs(expanded_graph)

        # detect aggregation queries
        is_aggregate = self.llm_node_selector.detect_aggregation_query(question)
        if is_aggregate:
            # table-level aggregation
            selected_rows = self.llm_node_selector.aggregate_column_wise(question, table_id_to_row_id_to_linked_passage_ids)

            if len(selected_rows) != 0:
                for table_id, row_id, linked_passage_ids in selected_rows:
                    table_segment_id = f"{table_id}_{row_id}"

                    if table_segment_id not in [bipartite_subgraph_candidate['table_segment_id'] for bipartite_subgraph_candidate in star_graph_list]:

                        if len(linked_passage_ids) == 0:
                            try:
                                linked_passage_titles = []
                                linked_passage_ids = self.table_segment_id_to_linked_passage_ids[table_segment_id]
                                linked_passage_titles = linked_passage_ids
                            except:
                                linked_passage_titles = []

                        star_graph_list.append({"table_segment_id":table_segment_id, "linked_passage_ids": linked_passage_titles})

        filtered_bipartite_subgraph_candidate_list = []
        for star_graph in star_graph_list:

            if len(star_graph['linked_passage_ids']) < 1:
                continue

            filtered_bipartite_subgraph_candidate_list.append(star_graph)

        # passage verification
        table_segment_id_to_linked_passage_id_list = self.llm_node_selector.verify_passages(question, filtered_bipartite_subgraph_candidate_list)

        for table_segment_id, linked_passage_id_list in table_segment_id_to_linked_passage_id_list.items():
            for passage_id in linked_passage_id_list:
                self.add_node(expanded_graph, 'table segment', table_segment_id, passage_id, 1000000, 'llm_selected')
                self.add_node(expanded_graph, 'passage', passage_id, table_segment_id, 1000000, 'llm_selected')
        
        return expanded_graph

    def add_node(self, graph, source_node_type, source_node_id, target_node_id, score, retrieval_type, source_rank = 0, target_rank = 0):
        if source_node_id not in graph:
            graph[source_node_id] = {'type': source_node_type, 'linked_nodes': [[target_node_id, score, retrieval_type, source_rank, target_rank]]}
        else:
            graph[source_node_id]['linked_nodes'].append([target_node_id, score, retrieval_type, source_rank, target_rank])

    def decompose_into_star_graphs(self, expanded_graph):
        table_segment_id_to_linked_passage_ids = {}
        table_id_to_row_id_to_linked_passage_ids = {}
        star_graph_list = []

        for node_id, node_info in expanded_graph.items():
            if node_info['type'] == 'table segment':
                linked_passage_scores = {}
                linked_nodes = sorted(node_info['linked_nodes'], key=lambda x: x[1], reverse=True)
                for linked_node_info in linked_nodes:
                    passage_id = linked_node_info[0]
                    score = linked_node_info[1]
                    if passage_id not in linked_passage_scores or score > linked_passage_scores[passage_id]:
                        linked_passage_scores[passage_id] = score
                table_segment_id_to_linked_passage_ids[node_id] = linked_passage_scores

        for table_segment_id, linked_passage_scores in table_segment_id_to_linked_passage_ids.items():
            table_id = table_segment_id.split('_')[0]
            row_id = table_segment_id.split('_')[1]
            if table_id not in table_id_to_row_id_to_linked_passage_ids:
                table_id_to_row_id_to_linked_passage_ids[table_id] = {}
            
            if row_id not in table_id_to_row_id_to_linked_passage_ids[table_id]:
                table_id_to_row_id_to_linked_passage_ids[table_id][row_id] = {}
            
            for passage_id, score in linked_passage_scores.items():
                if passage_id not in table_id_to_row_id_to_linked_passage_ids[table_id][row_id] or score > table_id_to_row_id_to_linked_passage_ids[table_id][row_id][passage_id]:
                    table_id_to_row_id_to_linked_passage_ids[table_id][row_id][passage_id] = score

            passage_scores = table_id_to_row_id_to_linked_passage_ids[table_id][row_id]
            sorted_passage_ids = [passage_id for passage_id, score in sorted(passage_scores.items(), key=lambda item: item[1], reverse=True)]
            star_graph_list.append(
                {
                    "table_segment_id": table_segment_id, 
                    "linked_passage_ids": sorted_passage_ids
                }
            )

            
        table_id_list = list(table_id_to_row_id_to_linked_passage_ids.keys())
        
        for table_id in table_id_list:
            row_id_to_linked_passage_ids = table_id_to_row_id_to_linked_passage_ids[table_id]
            table_content = self.table_key_to_content[table_id]
            rows = table_content['text'].split('\n')[1:]
            for row_id, row in enumerate(rows):
                if row == "":
                    continue
                if str(row_id) not in row_id_to_linked_passage_ids:
                    row_id_to_linked_passage_ids[str(row_id)] = []
                    
        
        return star_graph_list, table_id_to_row_id_to_linked_passage_ids
    
if __name__ == "__main__":
    main()