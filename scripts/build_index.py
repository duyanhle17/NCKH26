from src.graphrag_build.config import BuildConfig
from src.graphrag_build.pipeline import run_build

if __name__ == "__main__":
    cfg = BuildConfig(
        dataset_dir="./dataset",
        cache_dir="./artifacts/graphrag_bge"
    )
    run_build(cfg)