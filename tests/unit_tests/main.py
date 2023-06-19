import json
import os
from typing import Any
from langchain.llms.loading import load_llm

from langchain.llms.azureml_endpoint import (
    AzureMLOnlineEndpoint,
    ContentFormatterBase,
    OSSContentFormatter,
    AzureMLEndpointClient,
)


class CustomFormatter(ContentFormatterBase):
    content_type = "application/json"
    accepts = "application/json"

    def format_request_payload(self, prompt, model_kwargs) -> bytes:
        input_str = json.dumps(
            {
                "input_data": {
                    "columns": ["input_string"],
                    "index": [0],
                    "data": [prompt],
                }
            }
        )
        return str.encode(input_str)

    def format_response_payload(self, output) -> Any:
        response_json = json.loads(output)
        return response_json[0]["0"]


# llama_llm = AzureMLModel(
#     endpoint_url="https://openlm-llama-7b-700bt.eastus.inference.ml.azure.com/score",
#     endpoint_api_key=os.getenv("LLAMA_ENDPOINT_API_KEY"),
#     deployment_name="openlm-research-open-llama-7b--1",
#     model_kwargs={"temperature": 0, "max_tokens": 1000},
#     content_formatter=CustomFormatter(),
# )

azure_llm = AzureMLOnlineEndpoint(
    endpoint_url=os.getenv("OSS_ENDPOINT_URL"),
    endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),
    deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),
    model_kwargs={"temperature": 0.4, "top_p": 0.2, "max_tokens": 100}
    # content_formatter=OSSContentFormatter(),
)

azure_llm.save("azure_llm.json")
loaded = load_llm("azure_llm.json")
loaded.endpoint_url = os.getenv("OSS_ENDPOINT_URL")
loaded.endpoint_api_key = os.getenv("OSS_ENDPOINT_API_KEY")
loaded.content_formatter = OSSContentFormatter()
loaded.http_client = AzureMLEndpointClient(
    loaded.endpoint_url, loaded.endpoint_api_key, loaded.deployment_name
)
print(loaded("Foo"))
# print(new_llm("Write an essay on flowers. Essay: "))
# resp = azure_llm("Question: What is the color of the sky? Answer: ")
# resp = azure_llm("Write an essay on flowers.")
# print("Response: ", resp)
