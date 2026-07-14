import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class Settings:
    """应用配置，从环境变量加载。"""

    def __init__(self):
        self.dashscope_api_key: str = os.environ["DASHSCOPE_API_KEY"]
        self.tavily_api_key: str = os.environ["TAVILY_API_KEY"]
        self.llm_model: str = os.environ.get("LLM_MODEL", "qwen3.5-plus")
        self.llm_base_url: str = os.environ.get(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.database_url: str = os.environ.get("DATABASE_URL", "checkpoint.db")
        self.oss_access_key_id: str = os.environ["ALI_OSS_ACCESS_KEY_ID"]
        self.oss_access_key_secret: str = os.environ["ALI_OSS_ACCESS_KEY_SECRET"]
        self.oss_bucket_name: str = os.environ["ALI_OSS_BUCKET_NAME"]
        self.oss_endpoint: str = os.environ["ALI_OSS_ENDPOINT"]


settings = Settings()
