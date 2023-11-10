import logging
import sys

from typing import Dict
from llm_gateway_client.models.chat_generation_request import ChatGenerationRequest
from llm_gateway_client.models.conversational_memory_settings import ConversationalMemorySettings
from llm_gateway_client.models.generation_request import GenerationRequest
from llm_gateway_client.models.generation_response import GenerationResponse
from llm_gateway_client.models.chat_generation_response import ChatGenerationResponse
from openai.openai_object import OpenAIObject

# Initialize/configure logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter("[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _ch = logging.StreamHandler(stream=sys.stdout)
    _ch.setFormatter(formatter)
    logger.addHandler(_ch)


def as_chat_generation_request(config: Dict) -> ChatGenerationRequest:
    """Saturate an LLM Gateway `ChatGenerationRequest` object with a completion config.

    The completion config is assumed to be an OpenAI chat completion payload, for example:

        ```python
        {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Hello!"
                }
            ],
            "temperature": 0.8,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 256,
            "logit_bias": None,
            "n": 1,
            "response_format": { "type": "text" },
            "seed": 42,
            "stop": None,
            "stream": False,
            "top_p": 1,
            "tools": [],
            "tool_choice": "none",
            "user": None,
        }
        ```

    An equivalent LLM Gateway `ChatGenerationRequest` object is structured as follows:

        ```python
        {
            "user_message": "Hello!",
            "generation_settings": {
                "num_generations": 1,
                "max_tokens": 256,
                "temperature": 0.8,
                "stop_sequences": None,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "parameters": {
                    "model": "gpt-3.5-turbo",
                    "logit_bias": None,
                    "response_format": { "type": "text" },
                    "seed": 42,
                    "top_p": 1,
                    "tools": [],
                    "tool_choice": "none",
                    "user": None
                }
            },
            "conversational_memory_settings": {
                "strategy": "last_n_turns",
                "max_turns": 5
            }
        }
        ```

    Because the `Completion` class will call the chat generation endpoint if
    the specified model (eg. "gpt-3.5-turbo") supports chat completion, we may
    need to modify the input config. This happens in the `_construct_params`
    method in `Completion`, which gets called before this method. Here, we
    assume the input is consistent with a chat completion endpoint, and raise
    an error if the config suggests otherwise.
    """
    config_copy = config.copy()

    # sound the alarm if we got a completion config
    if "prompt" in config and "messages" not in config:
        raise ValueError("expected 'messages' key but found 'prompt'")

    # try to extract the content of the last user message, default to the empty string
    user_message = next(
        (msg.get("content") for msg in reversed(config_copy.pop("messages")) if msg.get("role") == "user"), ""
    )

    # TODO port all OpenAI settings to 'parameters'
    generation_settings = {
        "num_generations": config_copy.get("n", 1),
        "max_tokens": config_copy.get("max_tokens", 16),
        "temperature": config_copy.get("temperature", 0.5),
        "stop_sequences": config_copy.get("stop"),
        "frequency_penalty": config_copy.get("frequency_penalty", 0.0),
        "presence_penalty": config_copy.get("presence_penalty", 0.0),
        "parameters": {"model": config_copy.pop("model")},
    }
    conversational_memory_settings = ConversationalMemorySettings.from_dict({})

    return ChatGenerationRequest.from_dict(
        {
            "user_message": user_message,
            "generation_settings": generation_settings,
            "conversational_memory_settings": conversational_memory_settings,
        }
    )


def as_generation_request(config: Dict) -> GenerationRequest:
    """Saturate an LLM Gateway `GenerationRequest` object with a completion config.

    The completion config is assumed to be an OpenAI completion payload, for example:

        ```python
        {
            "model": "text-davinci-003",
            "prompt": "Write a SFW joke about the following topic: vegetables",
            "temperature": 0.8,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 256,
            "logit_bias": None,
            "logprobs": None,
            "n": 1,
            "response_format": { "type": "text" },
            "seed": 42,
            "stop": None,
            "stream": False,
            "suffix": None,
            "top_p": 1,
            "tools": [],
            "tool_choice": "none",
            "user": None,
        }
        ```

    An equivalent LLM Gateway `GenerationRequest` object is structured as follows:

        ```python
        {
            "model": "text-davinci-003",
            "prompt": "Write a SFW joke about the following topic: vegetables",
            "num_generations": 1,
            "enable_pii_masking": None,
            "temperature": 0.8,
            "stop_sequences": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 256,
            "parameters": {
                "logit_bias": None,
                "logprobs": None,
                "n": 1,
                "response_format": { "type": "text" },
                "seed": 42,
                "stream": False,
                "suffix": None,
                "top_p": 1,
                "tools": [],
                "tool_choice": "none",
                "user": None,
            }
        }
        ```

    Because the `Completion` class will call the chat generation endpoint if
    the specified model (eg. "gpt-3.5-turbo") supports chat completion, we may
    need to modify the input config. This happens in the `_construct_params`
    method in `Completion`, which gets called before this method. Here, we
    assume the input is consistent with a standard completion endpoint, and
    raise an error if the config suggests otherwise.
    """
    config_copy = config.copy()

    # sound the alarm if we got a chat completion config
    if "messages" in config and not "prompt" in config:
        raise ValueError("expected 'prompt' key but found 'messages'")

    # TODO port all OpenAI settings to 'parameters'
    generation_settings = {
        "prompt": config_copy.get("prompt"),
        "num_generations": config_copy.get("n", 1),
        "max_tokens": config_copy.get("max_tokens"),
        "enable_pii_masking": config_copy.get("enable_pii_masking", False),
        "temperature": config_copy.get("temperature", 0.5),
        "stop_sequences": config_copy.get("stop"),
        "frequency_penalty": config_copy.get("frequency_penalty", 0.0),
        "presence_penalty": config_copy.get("presence_penalty", 0.0),
        "model": config_copy.get("model"),
        "parameters": None,
    }

    return GenerationRequest.from_dict(generation_settings)


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
