from langchain.llms.azureml_endpoint import LLMBodyHandler, AzureMLModel
from pydantic import ValidationError

import urllib
import pytest
import json
import os

def test_oss_call() -> None:
    """Test valid call to Open Source Foundation Model"""
    class BodyHandler(LLMBodyHandler):
                content_type = "application/json"
                accepts = "application/json"
                
                def format_request_payload(self, prompt, model_kwargs) -> bytes:
                    input_str = json.dumps({"inputs": {"input_string": [prompt]}, "parameters": model_kwargs})
                    return str.encode(input_str)

                def format_response_payload(self, output) -> str:
                    response_json = json.loads(output)
                    return response_json[0]["0"]
    
    llm = AzureMLModel(
        endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),
        endpoint_url=os.getenv("OSS_ENDPOINT_URL"),
        deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),
        body_handler=BodyHandler()
    )
    output = llm("Foo")
    assert isinstance(output, str)
    

def test_hf_call() -> None:
    """Test valid call to HuggingFace Foundation Model"""
    class BodyHandler(LLMBodyHandler):
                content_type = "application/json"
                accepts = "application/json"
                
                def format_request_payload(self, prompt, model_kwargs) -> bytes:
                    input_str = json.dumps({"inputs":  [prompt], "parameters": model_kwargs})
                    return str.encode(input_str)

                def format_response_payload(self, output) -> str:
                    response_json = json.loads(output)
                    return response_json[0][0]["generated_text"]
    
    llm = AzureMLModel(
        endpoint_api_key=os.getenv("HF_ENDPOINT_API_KEY"),
        endpoint_url=os.getenv("HF_ENDPOINT_URL"),
        deployment_name=os.getenv("HF_DEPLOYMENT_NAME"),
        body_handler=BodyHandler()
    )
    output = llm("Foo")
    assert isinstance(output, str)

def test_dolly_call() -> None:
    """Test valid call to dolly-v2-12b"""
    class BodyHandler(LLMBodyHandler):
                content_type = "application/json"
                accepts = "application/json"
                
                def format_request_payload(self, prompt, model_kwargs) -> bytes:
                    input_str = json.dumps({"input_data": {"input_string": [prompt]}, "parameters": model_kwargs})
                    return str.encode(input_str)

                def format_response_payload(self, output) -> str:
                    response_json = json.loads(output)
                    return response_json[0]
    
    llm = AzureMLModel(
        endpoint_api_key=os.getenv("DOLLY_ENDPOINT_API_KEY"),
        endpoint_url=os.getenv("DOLLY_ENDPOINT_URL"),
        deployment_name=os.getenv("DOLLY_DEPLOYMENT_NAME"),
        body_handler=BodyHandler()
    )
    output = llm("Foo")
    assert isinstance(output, str)

def test_missing_body_handler() -> None:
    with pytest.raises(ValidationError):
        AzureMLModel(
            endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),
            endpoint_url=os.getenv("OSS_ENDPOINT_URL"),
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME")
        )

def test_invalid_request_format() -> None:
    class BodyHandler(LLMBodyHandler):
                content_type = "application/json"
                accepts = "application/json"
                
                def format_request_payload(self, prompt, model_kwargs) -> bytes:
                    input_str = json.dumps({"incorrect_input": {"input_string": [prompt]}, "parameters": model_kwargs})
                    return str.encode(input_str)

                def format_response_payload(self, output) -> str:
                    response_json = json.loads(output)
                    return response_json[0]["0"]
                
    with pytest.raises(json.JSONDecodeError):
      llm = AzureMLModel(
            endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),
            endpoint_url=os.getenv("OSS_ENDPOINT_URL"),
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),
            body_handler=BodyHandler()  
      )
      llm("Foo")

