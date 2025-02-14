import concurrent.futures
from FlagEmbedding import LayerWiseFlagLLMReranker

class CrossReranker:
    def __init__(self, 
                 checkpoint_path, 
                 process_num = 2, 
                 cutoff_layer = 28,
                ):
        self.process_num = process_num
        self.reranker_list = []
        for i in range(self.process_num):
            print(f"Loading reranker {i}...")
            reranker = LayerWiseFlagLLMReranker(checkpoint_path, use_fp16=True, device=f'cuda:{i}')
            print(f"Loading reranker {i} complete!")
            self.reranker_list.append(reranker)

        self.cutoff_layer = cutoff_layer

    def rerank(self, model_input, max_length):
        devided_model_input = []
        for rank in range(self.process_num):
            devided_model_input.append(model_input[rank*len(model_input)//self.process_num:(rank+1)*len(model_input)//self.process_num])

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_list = []
            reranking_scores = []
            
            for rank in range(self.process_num):
                future = executor.submit(self.rerank_worker, (self.reranker_list[rank], devided_model_input[rank], max_length))
                future_list.append(future)
            
            concurrent.futures.wait(future_list)
            for future in future_list:
                reranking_scores.extend(future.result())
        
        return model_input, reranking_scores

    def rerank_worker(self, reranker_and_model_input):
        reranker = reranker_and_model_input[0]
        model_input = reranker_and_model_input[1]
        max_length = reranker_and_model_input[2]
        
        if model_input == []:
            return []
        
        reranking_scores = reranker.compute_score(model_input, batch_size=60, cutoff_layers=[self.cutoff_layer], max_length=max_length)
        
        if len(model_input) == 1:
            reranking_scores = [reranking_scores]
        
        return reranking_scores