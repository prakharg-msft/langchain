from langchain.llms.azureml_endpoint import AzureMLModel, LLMContentFormatter
from langchain.llms.loading import load_llm
import json
import os

class OSSBodyHandler(LLMContentFormatter):
    content_type = "application/json"
    accepts = "application/json"
    
    def format_request_payload(self, prompt, model_kwargs) -> bytes:
        input_str = json.dumps({"inputs": {"input_string": [prompt]}, "parameters": model_kwargs})
        return str.encode(input_str)

    def format_response_payload(self, output) -> str:
        response_json = json.loads(output)
        return response_json[0]["0"]

azure_llm = AzureMLModel(
    endpoint_url=os.getenv("OSS_ENDPOINT_URL"),
    endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),
    deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),
    model_kwargs={"temperature": 0.8},
    body_handler=OSSBodyHandler()
)

# resp = azure_llm("Why is the sky blue?")
# print(resp)