"""Test AzureML Endpoint wrapper."""

from langchain.llms.azureml_endpoint import LLMContentFormatter, AzureMLModel, OSSContentFormatter, HFContentFormatter, DollyContentFormatter
from langchain.llms.loading import load_llm

import pytest
import json
import os
    
def test_oss_call() -> None:
    """Test valid call to Open Source Foundation Model."""
    llm = AzureMLModel(
        endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),
        endpoint_url=os.getenv("OSS_ENDPOINT_URL"),
        deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),
        content_formatter=OSSContentFormatter()
    )
    output = llm("Foo")
    assert isinstance(output, str)

def test_hf_call() -> None:
    """Test valid call to HuggingFace Foundation Model."""
    llm = AzureMLModel(
        endpoint_api_key=os.getenv("HF_ENDPOINT_API_KEY"),
        endpoint_url=os.getenv("HF_ENDPOINT_URL"),
        deployment_name=os.getenv("HF_DEPLOYMENT_NAME"),
        content_formatter=HFContentFormatter()
    )
    output = llm("Foo")
    assert isinstance(output, str)

def test_dolly_call() -> None:
    """Test valid call to dolly-v2-12b."""
    llm = AzureMLModel(
        endpoint_api_key=os.getenv("DOLLY_ENDPOINT_API_KEY"),
        endpoint_url=os.getenv("DOLLY_ENDPOINT_URL"),
        deployment_name=os.getenv("DOLLY_DEPLOYMENT_NAME"),
        content_formatter=DollyContentFormatter()
    )
    output = llm("Foo")
    assert isinstance(output, str)

def test_custom_formatter() -> None:
    """Test ability to create a custom content formatter."""
    class CustomFormatter(LLMContentFormatter):
        content_type = "application/json"
        accepts = "application/json"

        def format_request_payload(self, prompt, model_kwargs) -> bytes:
            input_str = json.dumps({"inputs": [prompt], "parameters": model_kwargs, "options": {"use_cache": False, "wait_for_model": True}})
            return input_str.encode("utf-8")

        def format_response_payload(self, output) -> bytes:
            response_json = json.loads(output)
            return response_json[0][0]["summary_text"]
        
    llm = AzureMLModel(
        endpoint_api_key=os.getenv("BERT_ENDPOINT_API_KEY"),
        endpoint_url=os.getenv("BERT_ENDPOINT_URL"),
        deployment_name=os.getenv("BERT_DEPLOYMENT_NAME"),
        content_formatter=CustomFormatter()
    )
    output = llm("Foo")
    assert isinstance(output, str)

def test_missing_body_handler() -> None:
    """Test AzureML LLM without a body_handler attribute"""
    with pytest.raises(AttributeError):
        llm = AzureMLModel(
            endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),
            endpoint_url=os.getenv("OSS_ENDPOINT_URL"),
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME")
        )
        llm("Foo")

def test_invalid_request_format() -> None:
    """Test invalid request format."""
    class CustomContentFormatter(LLMContentFormatter):
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
            content_formatter=CustomContentFormatter()  
      )
      llm("Foo")

def test_saving_loading_llm(tmp_path) -> None:
    """Test saving/loading an AzureML Foundation Model LLM."""
    
    llm = AzureMLModel(
        model_kwargs={"temperature": 0.03, "top_p": 0.4, "max_tokens": 200}
    )
    llm.save(file_path=tmp_path / "azureml.yaml")
    loaded_llm = load_llm(tmp_path / "azureml.yaml")

    assert loaded_llm == llm