from typing import List
import pandas as pd
from src.Agent.schemas import NewsArticle

# 프롬프트 로드 함수
def load_prompts(name : str):
    import yaml
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    agent_dir = os.path.dirname(current_dir)
    prompts_path = os.path.join(agent_dir, "prompts", "prompts.yaml")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

        prompt = prompts[name]
        return prompt

def format_news_articles(news_articles : List[NewsArticle]) -> str:
    """
    NewsArticle 형식의 뉴스 기사들을 LLM이 이해할 수 있는 문자열 하나로 변환
    DataFrame 형식의 표는 Markdown 형식으로 변환
    """
    formatted_text = ""

    for idx, article in enumerate(news_articles, 1):
        formatted_text += f"=== Article {idx} ===\n"
        formatted_text += f"Title: {article.title}\n"
        formatted_text += f"Editor: {article.editor}\n"
        formatted_text += f"Date: {article.date}\n"
        formatted_text += f"Content: \n"

        for block in article.content:
            if isinstance(block, pd.DataFrame):
                try:
                    table_md = block.to_markdown(index = False)
                    formatted_text += f"\n[Table Data]\n{table_md}\n\n"
                except ImportError:
                    formatted_text += f"\n[Table Data]\n{block.to_string()}\n\n"
            else:
                formatted_text += f"{str(block)}\n"
        
        formatted_text += "\n" + "=" * 30 + "\n\n"
    
    return formatted_text