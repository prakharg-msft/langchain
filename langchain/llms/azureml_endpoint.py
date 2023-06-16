"""Wrapper around AzureML Managed Online Endpoint API."""
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional
import urllib.request

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from pydantic import BaseModel, validator
import json

#TODO: Use python SDK instead of urllib
class AzureMLEndpointClient(object):
    """Wrapper around AzureML Managed Online Endpoint Client."""

    def __init__(self, endpoint_url, endpoint_api_key, deployment_name):
        """Initialize the class."""
        self.endpoint_url = endpoint_url
        self.endpoint_api_key = endpoint_api_key
        self.deployment_name = deployment_name
    
    def call(self, body):
        """call."""

        url = self.endpoint_url
        # Replace this with the primary/secondary key or AMLToken for the endpoint
        api_key = self.endpoint_api_key
        if not api_key:
            raise Exception("A key should be provided to invoke the endpoint")

        # The azureml-model-deployment header will force the request to go to a specific deployment.
        # Remove this header to have the request observe the endpoint traffic rules

        headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key), 'azureml-model-deployment': self.deployment_name}
        
        req = urllib.request.Request(url, body, headers)
        try:
            response = urllib.request.urlopen(req, timeout=50)
            result = response.read()
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))
            result = str(error)
        except Exception as e:
            print("Calling Azure Managed Online endpoint failed!")
            result = str(e)
        return result

class ContentFormatterBase():
    """A handler class to transform input from LLM to
    a format that AzureML endpoint expects.
    """

    """
    Example:
        .. code-block:: python

            class ContentFormatter(ContentFormatterBase):
                content_type = "application/json"
                accepts = "application/json"
                
                def format_request_payload(self, prompt, model_kwargs) -> bytes:
                    input_str = json.dumps({"inputs": {"input_string": [prompt]}, "parameters": model_kwargs})
                    return str.encode(input_str)

                def format_response_payload(self, output) -> str:
                    response_json = json.loads(output)
                    return response_json[0]["0"]
    """
    content_type: Optional[str] = "text/plain"
    """The MIME type of the input data passed to the endpoint"""
    
    accepts: Optional[str] = "text/plain"
    """The MIME type of the response data returned form the endpoint"""

    @abstractmethod
    def format_request_payload(self, prompt, model_kwargs) -> bytes:
        """Formats the request body according to the input schema of 
        the model. Returns bytes or seekable file like object in the
        format specified in the content_type request header.
        """

    @abstractmethod
    def format_response_payload(self, output) -> Any:
        """Formats the response body according to the output
        schema of the model. Returns the data type that is
        received from the response.
        """
     
# class LLMContentFormatter(ContentFormatterBase):
#     """Content handler for LLM class."""

class OSSContentFormatter(ContentFormatterBase):
    """Content handler for LLMs from the OSS catalog."""
    content_type = "application/json"
    accepts = "application/json"
    
    def format_request_payload(self, prompt, model_kwargs) -> bytes:
        input_str = json.dumps({"inputs": {"input_string": [prompt]}, "parameters": model_kwargs})
        return str.encode(input_str)

    def format_response_payload(self, output) -> str:
        response_json = json.loads(output)
        return response_json[0]["0"]
    
class HFContentFormatter(ContentFormatterBase):
    """Content handler for LLMs from the HuggingFace catalog."""
    content_type = "application/json"
    accepts = "application/json"
    
    def format_request_payload(self, prompt, model_kwargs) -> bytes:
        input_str = json.dumps({"inputs":  [prompt], "parameters": model_kwargs})
        return str.encode(input_str)

    def format_response_payload(self, output) -> str:
        response_json = json.loads(output)
        return response_json[0][0]["generated_text"]

class DollyContentFormatter(ContentFormatterBase):
    """Content handler for the Dolly-v2-12b model"""
    content_type = "application/json"
    accepts = "application/json"
    
    def format_request_payload(self, prompt, model_kwargs) -> bytes:
        input_str = json.dumps({"input_data": {"input_string": [prompt]}, "parameters": model_kwargs})
        return str.encode(input_str)

    def format_response_payload(self, output) -> str:
        response_json = json.loads(output)
        return response_json[0]

class AzureMLModel(LLM, BaseModel):
    """Wrapper around Azure ML Hosted models using Managed Online Endpoints.

    Example:
        .. code-block:: python

            auzre_llm = AzureMLModel(
                endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/score",
                endpoint_api_key="my-api-key",
                deployment_name="my-deployment-name",
                content_formatter=content_formatter)
    """

    endpoint_url: str = None
    """ URL of prexisting Endpoint """
    
    endpoint_api_key: str = None
    """ Authentication Key for Endpoint"""
    
    deployment_name: str = None
    """ Deployment Name for Endpoint"""

    http_client: Any = None  #: :meta private:
    
    content_formatter: ContentFormatterBase = None
    """The body handler class that provides an input and output
    transform function to handle formats between the LLM and
    the endpoint"""


    """
        Example:
            .. code-block:: python

            from langchain.llms.azureml_endpoint import LLMBodyHandler

            class BodyHandler(LLMBodyHandler):
                content_type = "application/json"
                accepts = "application/json"
                
                def format_request_payload(self, prompt, model_kwargs) -> bytes:
                    input_str = json.dumps({"inputs": {"input_string": [prompt]}, "parameters": model_kwargs})
                    return str.encode(input_str)

                def format_response_payload(self, output) -> str:
                    response_json = json.loads(output)
                    return response_json[0]["0"]
                    
    """

    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    @validator("http_client", always=True, allow_reuse=True)
    @classmethod
    def validate_client(cls, field_value, values) -> Dict:
        """Validate that api key and python package exists in environment."""
        endpoint_key = get_from_dict_or_env(
            values, "endpoint_api_key", "ENDPOINT_API_KEY"
        )
        endpoint_url = values["endpoint_url"]
        deployment_name = values["deployment_name"]
        http_client = AzureMLEndpointClient(endpoint_url, endpoint_key, deployment_name)
        return http_client

    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"deployment_name": self.deployment_name},
            **{"model_kwargs": _model_kwargs}
        }
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "azureml_endpoint"

    def _call(
        self,
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call out to an AzureML Managed Online endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = azureml_model("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}

        body = self.content_formatter.format_request_payload(prompt, _model_kwargs)
        endpoint_response = self.http_client.call(body)
        response = self.content_formatter.format_response_payload(endpoint_response)
        # TODO: Add error handling
        return response