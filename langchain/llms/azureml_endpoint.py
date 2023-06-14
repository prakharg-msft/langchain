"""Wrapper around AzureML Managed Online Endpoint API."""
from typing import Any, Dict, List, Mapping, Optional
import urllib.request

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from pydantic import BaseModel, validator
import json

OPEN_SOURCE = "open_source"
HUGGING_FACE = "hugging_face"

#TODO: Use python SDK instead of urllib
class AzureMLEndpointClient(object):
    """Wrapper around AzureML Managed Online Endpoint Client."""

    def __init__(self, endpoint_url, endpoint_api_key, deployment_name):
        """Initialize the class."""
        self.endpoint_url = endpoint_url
        self.endpoint_api_key = endpoint_api_key
        self.deployment_name = deployment_name
    
    def call(self, data):
        """call."""
        
        body = str.encode(json.dumps(data))

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


class AzureMLModel(LLM, BaseModel):
    """Wrapper around Azure ML Hosted models using Managed Online Endpoints.

    Example:
        .. code-block:: python

            auzre_llm = AzureMLModel(
                endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/score",
                endpoint_api_key="my-api-key",
                deployment_name="my-deployment-name",
                catalog_type="my-catalog-type")
    """

    endpoint_url: str = None
    """ URL of prexisting Endpoint """
    
    endpoint_api_key: str = None
    """ Authentication Key for Endpoint"""
    
    deployment_name: str = None
    """ Deployment Name for Endpoint"""

    catalog_type: str = None
    """ Model Catalog Type: hugging_face or open_source """

    http_client: Any = None  #: :meta private:
    
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
        catalog_type = values["catalog_type"]
        http_client = AzureMLEndpointClient(endpoint_url, endpoint_key, deployment_name)
        return http_client

    def format_body(self, prompt, parameters) -> Mapping[str, Any]:
        """Format the body of the request according to the catalog type"""
        # TODO: Add support for dolly-v2-12b input format
        if self.catalog_type == OPEN_SOURCE:
            return {"inputs": {"input_string": [prompt]}, "parameters": parameters} 
        elif self.catalog_type == HUGGING_FACE:
            # HuggingFace default values for options
            options = {"use_cache": True, "wait_for_model": False}
            if "use_cache" in parameters:
                options["use_cache"] = parameters["use_cache"]
                parameters.pop("use_cache")
            if "wait_for_model" in parameters:
                options["wait_for_model"] = parameters["wait_for_model"]
                parameters.pop("wait_for_model")
            return {"inputs": [prompt], "parameters": parameters, "options": options}
        else:
            return None
        
    def format_response(self, response) -> str:
        """Get the first response"""
        # TODO: Add support for dolly-v2-12b output
        responses = response[0] if self.catalog_type == OPEN_SOURCE else response[0][0]
        for _, resp in responses.items():
            return resp
        
        return None
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"deployment_name": self.deployment_name},
            **{"model_kwargs": _model_kwargs},
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

        body = self.format_body(prompt, _model_kwargs)
        endpoint_response = self.http_client.call(body)
        response = self.format_response(json.loads(endpoint_response))
        # TODO: Add error handling
        return response
