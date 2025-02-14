import json
from ColBERT.colbert import Searcher
from ColBERT.colbert.infra import ColBERTConfig
from Ours.utils.utils import read_jsonl, disablePrint, enablePrint

class ColBERTRetriever:
    def __init__(self,
                 index_name,
                 ids_path, 
                 collection_path, 
                 index_root_path, 
                 checkpoint_path
                ):

        print("Loading id mappings...")
        self.id_to_key = json.load(open(ids_path))
        print("Loaded id mappings!")

        print("Loading index...")
        disablePrint()
        self.searcher = Searcher(index=index_name, config=ColBERTConfig(), collection=collection_path, index_root=index_root_path, checkpoint=checkpoint_path)
        enablePrint()
        print("Loaded index complete!")

    def search(self, query, k=10000):
        retrieved_info = self.searcher.search(query, k = k)
        retrieved_id_list = retrieved_info[0]
        retrieved_score_list = retrieved_info[2]
        
        retrieved_key_list = [self.id_to_key[str(id)] for id in retrieved_id_list]
        
        return retrieved_key_list, retrieved_score_list