from langchain.llms.azureml_endpoint import AzureMLModel
import os
azure_llm = AzureMLModel(
    endpoint_url=os.getenv("ENDPOINT_URL"),
    endpoint_api_key=os.getenv("ENDPOINT_API_KEY")
)

resp = azure_llm(prompt="HI! HOW ARE YOU??")
print("Response:", resp)