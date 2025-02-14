import logging
import argparse
from typing import Dict
from flask import Flask, request
from flask_cors import CORS
from waitress import serve
from colbert_retriever import ColBERTRetriever

# Initialize argument parser
parser = argparse.ArgumentParser(description="Run the Flask app with custom index and ids paths")
parser.add_argument('--index_name', type=str, required=True, help='Name of the index to use')
parser.add_argument('--ids_path', type=str, required=True, help='Path to the ids')
parser.add_argument('--collection_path', type=str, required=True, help='Path to the collection')
parser.add_argument('--index_root_path', type=str, required=True, help='Path to the index root')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint')

args = parser.parse_args()

# Initialize Flask
app = Flask(__name__)
cors = CORS(app)

# Initialize Index
index_name = args.index_name
ids_path = args.ids_path
collection_path = args.collection_path
index_root_path = args.index_root_path
checkpoint_path = args.checkpoint_path
retriever = ColBERTRetriever(f"{index_name}.nbits2", ids_path, collection_path, index_root_path, checkpoint_path)

@app.route("/table_segment_retrieve", methods=["GET", "POST", "OPTIONS"])
def table_segment_retrieve():
    # Handle GET and POST requests
    if request.method == "POST":
        params: Dict = request.json
    else:
        params: Dict = request.args.to_dict()

    query = params["query"]
    k = params.get("k", 10000)

    retrieved_key_list, retrieved_score_list = retriever.search(query, k=k)
    
    response = {"retrieved_key_list": retrieved_key_list, "retrieved_score_list": retrieved_score_list}

    return response

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    serve(app, host="0.0.0.0", port=5003)