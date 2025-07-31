import os
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.agent import Agent
from agno.team.team import Team
from dotenv import load_dotenv
from agno.models.openai import OpenAIChat
from agno.tools.tavily import TavilyTools
from agno.storage.postgres import PostgresStorage

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

AGENT_SESSIONS = "agent_sessions"
AGENT_MEMORIES = "agent_memories"
DB_URL = "postgresql+psycopg://postgres:2244@localhost:5432/sevensix_db"


tavily_tools = TavilyTools(api_key=tavily_api_key)
model = OpenAIChat(id="gpt-4o", api_key=openai_api_key)

knowledge_base = TextKnowledgeBase(
    vector_db=ChromaDb(collection="headphone_buying_guide", path="./vector_storage/"),
)

storage = PostgresStorage(table_name=AGENT_SESSIONS, db_url=DB_URL)


web_agent = Agent(
    name="Web Search Agent",
    role="Research Assistant for up-to-date headphone information",
    tools=[tavily_tools],
    model=model,
    knowledge=knowledge_base,
    instructions=[
        "You are activated only when the Guiding Agent needs current market information not available in the knowledge base.",
        "Your role is to supplement knowledge base information with:",
        "- Current prices and availability",
        "- Latest model releases",
        "- Recent expert reviews and user feedback",
        "- Comparison data between specific models",
        "**Search Strategy:**",
        "- Use precise, targeted search queries",
        "- Focus on recent information (within last 6-12 months)",
        "- Prioritize authoritative sources (tech review sites, manufacturer sites)",
        "- Synthesize findings into actionable insights",
        "**Output Format:**",
        "Provide concise, factual summaries that the Guiding Agent can use to enhance recommendations.",
    ],
)

summarizer_agent = Agent(
    name="Summarizer Agent",
    role="Information synthesizer",
    model=model,
    knowledge=knowledge_base,
    instructions=[
        "You condense information from web searches into clear, actionable insights.",
        "Focus on information most relevant to headphone recommendations:",
        "- Key product features and specifications",
        "- Pricing and value propositions",
        "- User experience highlights",
        "- Comparative advantages/disadvantages",
        "Keep summaries concise but comprehensive enough for recommendation decisions.",
    ],
)

email_agent = Agent(
    name="Email Agent",
    role="Generate and send email",
    model=model,
    instructions=[
        "You condense information from web searches into clear, actionable insights.",
        "Focus on information most relevant to headphone recommendations:",
        "- Key product features and specifications",
        "- Pricing and value propositions",
        "- User experience highlights",
        "- Comparative advantages/disadvantages",
        "Keep summaries concise but comprehensive enough for recommendation decisions.",
    ],
)
# Enhanced Team with better workflow control
headphone_guidance_team = Team(
    name="Headphone Recommendation System",
    mode="coordinate",
    model=model,
    user_id="100",
    team_id="101",
    session_id="102",
    storage=storage,
    instructions=[
        "You orchestrate a headphone recommendation system with a specific workflow:",
        "**Phase 1 - Information Gathering (Query Agent Focus):**",
        "- Always start by delegating to Query Agent for ALL user queries",
        "- Query Agent must continue asking questions until it declares 'QUERY COMPLETE'",
        "- Do NOT proceed to recommendations until Query Agent confirms all information is collected",
        "- If user tries to skip questions or asks for immediate recommendations, redirect them back to Query Agent",
        "**Phase 2 - Recommendation Generation:**",
        "- Only after 'QUERY COMPLETE' signal, delegate to Guiding Agent",
        "- If Guiding Agent needs current market data, activate Web Search Agent",
        "- Use Summarizer Agent to process any web search results",
        "- Return final recommendations to user",
        "**Phase 3 - Follow-up:**",
        "- Handle follow-up questions about recommendations through Guiding Agent",
        "- If user wants to change criteria, restart with Query Agent",
        "**Critical Rules:**",
        "- Never allow shortcuts in the information gathering phase",
        "- Maintain conversation flow and context between agent interactions",
        "- Ensure each agent completes its specific role before moving to the next",
        "- Keep the user informed about which phase they're in",
    ],
    members=[
        web_agent,
        summarizer_agent,
    ],
    add_datetime_to_instructions=True,
    markdown=True,
    debug_mode=True,
    show_members_responses=True,
)


def main():
    print("🎧 Welcome to the Headphone Recommendation System!")

    headphone_guidance_team.print_response(
        message="I need over-ear wireless headphones with ANC under $300. I prefer balanced sound with slight bass boost, maximum comfort with plush cushioning for 3-4 hour sessions, at least 25 hours battery life, touch controls and multipoint Bluetooth. Will use mainly for commuting on public transport and working from home. Design should be sleek and modern in black or dark colors.",
        stream=True,
        stream_intermediate_steps=True,
    )


if __name__ == "__main__":
    main()
