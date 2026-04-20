from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_chat_model: str = Field(default="llama3:latest", alias="OLLAMA_CHAT_MODEL")
    ollama_embed_model: str = Field(default="nomic-embed-text:latest", alias="OLLAMA_EMBED_MODEL")
    vector_db_path: str = Field(default="./rag_store", alias="VECTOR_DB_PATH")
    top_k: int = Field(default=4, alias="TOP_K")


settings = Settings()