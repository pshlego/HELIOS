import re
import json
import requests
import concurrent.futures
from transformers import AutoTokenizer
from models.prompt.prompts_v2 import detect_aggregation_query_prompt, select_row_wise_prompt, select_passages_prompt

class LlmNodeSelector:
    def __init__(self, cfg, table_key_to_content, passage_key_to_content):
        self.detect_aggregation_query_prompt = detect_aggregation_query_prompt
        self.select_row_wise_prompt = select_row_wise_prompt
        self.select_passages_prompt = select_passages_prompt
        self.table_key_to_content = table_key_to_content
        self.passage_key_to_content = passage_key_to_content
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.tokenizer = tokenizer
        self.llm_addr = "http://localhost:30000/generate"
        
    def detect_aggregation_query(self, query):
        prompt = self.generate_detect_aggregation_query_prompt(query)
        response_list = requests.post(
                self.llm_addr,
                json={
                    "text": [prompt],
                    "sampling_params": {
                        "max_new_tokens": 32,
                        "temperature": 0,
                    }
                },
                timeout=None,
            ).json()

        pattern_col = r"f_agg\(\[(.*?)\]\)"

        try:
            pred = re.findall(pattern_col, response_list[0]["text"], re.S)[0].strip()
        except:
            return False
        
        if 'true' == pred.replace("[", "").replace("]", "").lower():
            return True
        else:
            return False
            
    def aggregate_column_wise(self, query, table_id_to_row_id_to_linked_passage_ids):
        prompt_list = []
        table_id_list = []
        selected_rows = []
        for table_id, row_id_to_linked_passage_ids in table_id_to_row_id_to_linked_passage_ids.items():
            prompt = self.generate_select_row_wise_prompt(query, table_id, row_id_to_linked_passage_ids)
            prompt_list.append(prompt)
            table_id_list.append(table_id)

        response_list = requests.post(
                        self.llm_addr,
                        json={
                            "text": prompt_list,
                            "sampling_params": {
                                "max_new_tokens": 96,
                                "temperature": 0,
                            }
                        },
                        timeout=None,
                    ).json()

        pattern_col = r"f_row\(\[(.*?)\]\)"
        selected_table_id_to_row_id_list = {}
        for table_id, response in zip(table_id_list, response_list):
            try:
                pred = re.findall(pattern_col, response["text"], re.S)[0].strip()
            except Exception:
                continue
            
            selected_row_ids = pred.replace('"','').split(', ')
            selected_table_id_to_row_id_list[table_id] = []
            for i, selected_row_id in enumerate(selected_row_ids):
                try:
                    selected_table_id_to_row_id_list[table_id].append(int(selected_row_id.replace('row', '').strip()))
                except:
                    continue

        for table_id, row_id_list in selected_table_id_to_row_id_list.items():
            for row_id in row_id_list:
                if str(row_id) not in table_id_to_row_id_to_linked_passage_ids[str(table_id)]:
                    continue
                linked_passage_ids = table_id_to_row_id_to_linked_passage_ids[str(table_id)][str(row_id)]
                if linked_passage_ids != []:
                    linked_passage_ids = [passage_id for passage_id, score in sorted(linked_passage_ids.items(), key=lambda item: item[1], reverse=True)]
                selected_rows.append([table_id, row_id, linked_passage_ids])
        
        return selected_rows
        
    def verify_passages(self, question, bipartite_subgraph_candidate_list):
        prompt_list = []
        table_segment_id_to_passage_id_list = {}
        feeded_subgraph_candidate_list = []
        count_star_graph_original = 0
        count_star_graph_with_heuristic = 0
        for bipartite_subgraph_candidate in bipartite_subgraph_candidate_list:
            table_id = bipartite_subgraph_candidate['table_segment_id'].split('_')[0]
            row_id = bipartite_subgraph_candidate['table_segment_id'].split('_')[1]
            linked_passage_ids = bipartite_subgraph_candidate['linked_passage_ids']
            feeded_subgraph_candidate_list.append(bipartite_subgraph_candidate)
            prompt = self.generate_select_passages_prompt_v2(question, table_id, row_id, linked_passage_ids)
            count_star_graph_original += 1
            count_star_graph_with_heuristic += 1
            prompt_list.append(prompt)

        response_list = requests.post(
                        self.llm_addr,
                        json={
                            "text": prompt_list,
                            "sampling_params": {
                                "max_new_tokens": 160,
                                "temperature": 0,
                            }
                        },
                        timeout=None,
                    ).json()
        
        pattern_col = r"f_passage\(\[(.*?)\]\)"
        for bipartite_subgraph_candidate, response in zip(feeded_subgraph_candidate_list, response_list):
            try:
                pred = re.findall(pattern_col, response["text"].replace('"]) , "', '", "'), re.S)[0].strip()
            except Exception:
                continue
            
            if pred.replace('"','') in self.passage_key_to_content:
                pred_passage_list = [pred.replace('"','')]
            else:
                pred_passage_list = pred.split('", "')
                if len(pred_passage_list) == 1:
                    pred_passage_list = pred.replace('"','').split(", ")
            
            try:
                for i in range(len(pred.replace('"','').split(","))):
                    if ",".join(pred.replace('"','').split(",")[i:i+2]) in self.passage_key_to_content:
                        pred_passage_list.append(",".join(pred.replace('"','').split(",")[i:i+2]))
            except:
                pass
            
            selected_passage_list = []
            for pred_passage in pred_passage_list:
                if pred_passage.replace('"','') not in self.passage_key_to_content:
                    continue
                selected_passage_list.append(pred_passage.replace('"',''))

            table_segment_id = bipartite_subgraph_candidate['table_segment_id']
            if table_segment_id not in table_segment_id_to_passage_id_list:
                table_segment_id_to_passage_id_list[table_segment_id] = [] 
            table_segment_id_to_passage_id_list[table_segment_id].extend(list(set(selected_passage_list)))
            table_segment_id_to_passage_id_list[table_segment_id] = list(set(table_segment_id_to_passage_id_list[table_segment_id]))
        return table_segment_id_to_passage_id_list
    
    def generate_detect_aggregation_query_prompt(self, query):
        prompt = self.detect_aggregation_query_prompt.format(question=query)
        return prompt
    
    def generate_select_row_wise_prompt(self, query, table_id, row_id_to_linked_passage_ids):
        prompt = "" + self.select_row_wise_prompt.rstrip() + "\n\n"
        table_content = self.table_key_to_content[str(table_id)]
        table_title = table_content['title']
        table_text = table_content['text']
        column_names = table_text.split('\n')[0].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ').split(' | ')
        table_prompt_text = f"table caption : {table_title}\n"
        table_prompt_text += f"col : {column_names}\n"
        linked_passages_prompt_text = ""
        row_text_list = table_text.split('\n')[1:]
        row_text_list = [row_text for row_text in row_text_list if row_text != '']
        for row_id, row_text in enumerate(row_text_list):
            if str(row_id) not in row_id_to_linked_passage_ids:
                continue
            linked_passage_ids = row_id_to_linked_passage_ids[str(row_id)]
            table_prompt_text += f"row {row_id} : {row_text_list[int(row_id)].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ').split(' | ')}\n"
            linked_passages_prompt_text += f"passages linked to row {row_id}\n"
            for linked_passage_id in linked_passage_ids:
                passage_content = self.passage_key_to_content[linked_passage_id]
                linked_passage_text = passage_content['text']
                trimmed_text = self.trim(linked_passage_text, 32)
                linked_passages_prompt_text += f"Title: {passage_content['title']}. Content: {trimmed_text}\n"

        prompt = self.select_row_wise_prompt.format(question=query, table=table_prompt_text, linked_passages=linked_passages_prompt_text)
        return prompt

    def generate_select_passages_prompt_v2(self, question, table_id, row_id, linked_passage_ids):
        table_content = self.table_key_to_content[str(table_id)]
        table_title = table_content['title']
        table_text = table_content['text']
        column_names = table_text.split('\n')[0].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')
        row_values = table_text.split('\n')[1+int(row_id)].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')
        
        table_segment_text = f"table caption : {table_title}\n"
        table_segment_text += f"col : {column_names}\n"
        table_segment_text += f"row 1 : {row_values}"
    
        linked_passages_text = f"List of linked passages: {linked_passage_ids}\n"

        for linked_passage_id in linked_passage_ids:
            passage_content = self.passage_key_to_content[linked_passage_id]
            passage_text = passage_content['text']
            trimmed_text = self.trim(passage_text)
            linked_passages_text += f"\n\nTitle : {passage_content['title']}. Content: {trimmed_text}\n"
        
        prompt = self.select_passages_prompt.format(question=question, table_segment=table_segment_text, linked_passages=linked_passages_text)
        
        return prompt
    
    def trim(self, raw_text, trim_length=128):
        tokenized_text = self.tokenizer.encode(raw_text)
        trimmed_tokenized_text = tokenized_text[ : trim_length]
        trimmed_text = self.tokenizer.decode(trimmed_tokenized_text)
        trimmed_text = trimmed_text.replace('<|begin_of_text|>', '')
        return trimmed_text