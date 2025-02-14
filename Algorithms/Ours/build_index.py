import hydra
from omegaconf import DictConfig
from ColBERT.colbert import Indexer
from ColBERT.colbert.infra import Run, RunConfig, ColBERTConfig

@hydra.main(config_path="conf", config_name="build_index")
def main(cfg: DictConfig):
    with Run().context(RunConfig(nranks=4, experiment=cfg.index_name)):
        config = ColBERTConfig(
            nbits=2,
            root=cfg.collection_root_dir_path,
        )
        indexer = Indexer(checkpoint=cfg.colbert_checkpoint, config=config)
        indexer.index(name=f"{cfg.index_name}.nbits2", collection=cfg.collection_tsv_path, query_maxlen=cfg.query_maxlen)

if __name__ == "__main__":
    main()