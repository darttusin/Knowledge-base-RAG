from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    eval_embedding_model: str = "Snowflake/snowflake-arctic-embed-m"
    rerank_model: str = "BAAI/bge-reranker-base"

    llm_model_generation: str = "TechxGenus/c4ai-command-r-v01-AWQ"
    llm_api_url: str = ""
    llm_api_key: str = ""

    llm_model_judge: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    judge_api_url: str = ""
    judge_api_key: str = ""

    dataset_path: str = "./data/dataset"
    qa_dataset_path: str = "./data/stackoverflow-pytorch.csv"
    chroma_path: str = "./chroma_fast"
    chroma_collection: str = "docs_fast"

    wandb_project: str = "pytorch-rag-experiments"
    wandb_api_key: str = ""

    device: str = "cuda"
