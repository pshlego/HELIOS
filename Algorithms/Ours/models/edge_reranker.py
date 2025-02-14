import logging
import argparse
from typing import Dict
from flask import Flask, request
from flask_cors import CORS
from waitress import serve
from cross_reranker import CrossReranker

# Initialize argument parser
parser = argparse.ArgumentParser(description="Run the Flask app with custom index and ids paths")
parser.add_argument('--checkpoint_path', type=str, required=True, help='Name of the checkpoint')
parser.add_argument('--process_num', type=int, required=True, help='Number of processes')
parser.add_argument('--cutoff_layer', type=int, required=True, help='Cutoff layer')
args = parser.parse_args()  

# Initialize Flask
app = Flask(__name__)
cors = CORS(app)
checkpoint_path = args.checkpoint_path

# Initialize Edge Reranker
reranker = CrossReranker(checkpoint_path=checkpoint_path, process_num=args.process_num, cutoff_layer=args.cutoff_layer)

@app.route("/edge_rerank", methods=["GET", "POST", "OPTIONS"])
def rerank():
    # Handle GET and POST requests
    if request.method == "POST":
        params: Dict = request.json
    else:
        params: Dict = request.args.to_dict()
        
    model_input = params["model_input"]
    max_length = params["max_length"]
    model_input, reranking_scores = reranker.rerank(model_input, max_length)
    
    response = {"model_input": model_input, "reranking_scores": reranking_scores}
    
    return response

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    serve(app, host="0.0.0.0", port=5001)