import logging
import numpy as np
import os
import copy
from tqdm import tqdm
import csv
import torch
from functools import partial
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples

logger = logging.getLogger(__name__)

class CERerankingEvaluator:
    """
    This class evaluates a CrossEncoder model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 is compute to measure the quality of the ranking.

    :param samples: Must be a list and each element is of the form: {'query': '', 'positive': [], 'negative': []}. Query is the search query,
     positive is a list of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
    """
    def __init__(self, config, mrr_at_k: int = 10, name: str = '', write_csv: bool = True):
        dev_corpus = {}
        dev_collection_filepath = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Development_Dataset/edge/collection.tsv"
        with open(dev_collection_filepath, "r", encoding="utf8") as fIn:
            for line in fIn:
                pid, passage = line.strip().split("\t")
                dev_corpus[pid] = passage


        ### Read the train queries, store in queries dict
        dev_queries = {}
        dev_queries_filepath = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Development_Dataset/edge/queries.tsv"
        with open(dev_queries_filepath, "r", encoding="utf8") as fIn:
            for line in fIn:
                qid, query = line.strip().split("\t")
                dev_queries[qid] = query

        ### Now we create our training & dev data
        train_samples = []
        dev_samples = {}

        # We use 200 random queries from the train set for evaluation during training
        # Each query has at least one relevant and up to 200 irrelevant (negative) passages
        num_dev_queries = 2214 #2214
        num_max_dev_negatives = 200 #100

        # msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz and msmarco-qidpidtriples.rnd-shuf.train.tsv.gz is a randomly
        # shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website
        # We extracted in the train-eval split 500 random queries that can be used for evaluation during training
        train_eval_filepath = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Development_Dataset/edge/triples.tsv"

        with open(train_eval_filepath, "r", encoding="utf8") as fIn:
            for line in fIn:
                qid, pos_id, neg_id = line.strip().split("\t")

                if qid not in dev_samples and len(dev_samples) < num_dev_queries:
                    dev_samples[qid] = {"query": dev_queries[qid], "positive": set(), "negative": set()}

                if qid in dev_samples:
                    dev_samples[qid]["positive"].add(dev_corpus[pos_id])

                    if len(dev_samples[qid]["negative"]) < num_max_dev_negatives:
                        dev_samples[qid]["negative"].add(dev_corpus[neg_id])

        self.samples = dev_samples
        self.name = name
        self.mrr_at_k = mrr_at_k
        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        
        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        self.b_size = 64
        self.csv_file = "CERerankingEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MRR@{}".format(mrr_at_k)]
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CERerankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        all_mrr_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        for instance in tqdm(self.samples, desc="Evaluating"):
            query = instance['query']
            queries = self.query_tokenizer.tensorize([query])
            
            
            positive = list(instance['positive'])
            negative = list(instance['negative'])
            docs = positive + negative
            is_relevant = [True]*len(positive) + [False]*len(negative)

            if len(positive) == 0 or len(negative) == 0:
                continue

            num_queries += 1
            num_positives.append(len(positive))
            num_negatives.append(len(negative))
            #self.b_size
            total_scores = []
            queries = self.query_tokenizer.tensorize([query])
            for i in range(0, len(docs), self.b_size):
                docs_batch = docs[i:i+self.b_size]
                passages = self.doc_tokenizer.tensorize(docs_batch)
                Q = queries
                D = passages
                encoded_Q = model.module.query(*Q)
                encoded_D, encoded_D_mask = model.module.doc(*D, keep_dims='return_mask')

                # Repeat each query encoding for every corresponding document.
                Q_duplicated = encoded_Q.repeat_interleave(len(docs[i:i+self.b_size]), dim=0).contiguous()
                pred_scores = model.module.score(Q_duplicated, encoded_D, encoded_D_mask)
                total_scores.extend(pred_scores.detach().cpu().numpy().tolist())
                # torch.cuda.empty_cache()
            # passages = self.doc_tokenizer.tensorize(docs)
            # encoding = [queries, passages]
            # pred_scores = model(*encoding)
            pred_scores_argsort = np.argsort(-np.array(total_scores))  #Sort in decreasing order

            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0:self.mrr_at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank+1)
                    break

            all_mrr_scores.append(mrr_score)

        mean_mrr = np.mean(all_mrr_scores)
        logger.info("Queries: {} \t Positives: Min {:.1f}, Mean {:.1f}, Max {:.1f} \t Negatives: Min {:.1f}, Mean {:.1f}, Max {:.1f}".format(num_queries, np.min(num_positives), np.mean(num_positives), np.max(num_positives), np.min(num_negatives), np.mean(num_negatives), np.max(num_negatives)))
        logger.info("MRR@{}: {:.2f}".format(self.mrr_at_k, mean_mrr*100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_mrr])

        return mean_mrr