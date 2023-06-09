from langchain.llms.azureml_endpoint import AzureMLModel
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import os

azure_llm = AzureMLModel(
    endpoint_url=os.getenv("ENDPOINT_URL"),
    endpoint_api_key=os.getenv("ENDPOINT_API_KEY"),
    deployment_name="matthew-gpt-2"
)

search = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]
self_ask_with_search = initialize_agent(tools, azure_llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
self_ask_with_search.run("What is the hometown of the reigning men's U.S. Open champion?")
