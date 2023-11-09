import logging
import sys

from typing import Dict
from llm_gateway_client.models.chat_generation_request import ChatGenerationRequest
from llm_gateway_client.models.generation_request import GenerationRequest
from llm_gateway_client.models.generation_response import GenerationResponse
from llm_gateway_client.models.chat_generation_response import ChatGenerationResponse
from openai.openai_object import OpenAIObject


logger = logging.getLogger(__name__)
formatter = logging.Formatter("[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _ch = logging.StreamHandler(stream=sys.stdout)
    _ch.setFormatter(formatter)
    logger.addHandler(_ch)


def as_chat_generation_request(config: Dict) -> ChatGenerationRequest:
    return ChatGenerationRequest.from_dict(config)


def as_generation_request(config: Dict) -> GenerationRequest:
    return GenerationRequest.from_dict(config)


def convert_chat_generation_response(response: ChatGenerationResponse) -> OpenAIObject:
    """Convert a ChatGenerationResponse to an OpenAI object.

    This method creates a dictionary of entries from the LLM Gateway response
    to saturate an OpenAI object so that downstream methods for tuning/book
    keeping/etc. work as expected. Distinct from the
    'convert_generation_response' method mainly because of how chat messages
    are represented.

    See: https://platform.openai.com/docs/api-reference/chat/object
    """
    response_copy = response.copy()
    response_dict = response_copy.to_dict()
    try:
        openai_dict = {
            "id": response_dict["id"],
            "choices": [
                {
                    "index": msg["parameters"]["index"],
                    "finish_reason": msg["parameters"]["finish_reason"],
                    "message": {"role": msg["role"], "content": msg["content"]},
                }
                for msg in response_dict["generation_details"]["generations"]
            ],
        }

        openai_dict.update(response_dict.get("generation_details", {}).pop("parameters"))

        return OpenAIObject.construct_from(openai_dict)
    except Exception as e:
        logger.warning("Caught exception converting LLM Gateway response to OpenAI chat completion.")
        raise e


def convert_generation_response(response: GenerationResponse) -> OpenAIObject:
    """Convert a GenerationResponse to an OpenAI object.

    This method creates a dictionary of entries from the LLM Gateway response
    to saturate an OpenAI object so that downstream methods for tuning/book
    keeping/etc. work as expected.

    See: https://platform.openai.com/docs/api-reference/completions/object
    """
    response_copy = response.copy()
    response_dict = response_copy.to_dict()
    try:
        openai_dict = {
            "id": response_dict["id"],
            "choices": [{"text": gen["text"], **gen.pop("parameters")} for gen in response_dict["generations"]],
        }

        openai_dict.update(response_dict.pop("parameters"))

        return OpenAIObject.construct_from(openai_dict)
    except Exception as e:
        logger.warning("Caught exception converting LLM Gateway response to OpenAI completion.")
        raise e
