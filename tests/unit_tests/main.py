from typing import Any
from langchain.llms.azureml_endpoint import AzureMLModel, ContentFormatterBase
import json
import os
class CustomFormatter(ContentFormatterBase):
    content_type = "application/json"
    accepts = "application/json"

    def format_request_payload(self, prompt, model_kwargs) -> bytes:
        input_str = json.dumps({"input_data": {"columns": ["input_string"], "index": [0], "data": [prompt]}, "parameters": model_kwargs})
        return str.encode(input_str)
    
    def format_response_payload(self, output) -> Any:
        print(output)
        response_json = json.loads(output)
        return response_json[0]["0"]
    
azure_llm = AzureMLModel(
    endpoint_url="https://openlm-llama-7b-700bt.eastus.inference.ml.azure.com/score",
    endpoint_api_key=os.getenv("LLAMA_ENDPOINT_API_KEY"),
    deployment_name="openlm-research-open-llama-7b--1",
    model_kwargs={"temperature": 0.8, "max_tokens": 100},
    content_formatter=CustomFormatter()
)

resp = azure_llm("Why is the sky blue?")
print("Response: ", resp)