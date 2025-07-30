import os
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.agent import Agent
from dotenv import load_dotenv
from agno.models.openai import OpenAIChat

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. Please check your .env file."
    )

if not tavily_api_key:
    raise ValueError(
        "TAVILY_API_KEY not found in environment variables. Please check your .env file."
    )

model = OpenAIChat(id="gpt-4o", api_key=openai_api_key)

knowledge_base = TextKnowledgeBase(
    vector_db=ChromaDb(collection="headphone_buying_guide", path="./vector_storage/"),
)


agent = Agent(
    name="Guiding Agent",
    role="Guide to pick a headphone",
    model=OpenAIChat("gpt-4o"),
    knowledge=knowledge_base,
    instructions=[],
)


def main():
    pass


if __name__ == "__main__":
    main()
