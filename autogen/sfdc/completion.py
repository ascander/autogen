import asyncio

from dotenv import dotenv_values
from time import time
from typing import Dict

from .utils import as_generation_request, as_openai_response

from autogen.oai.completion import Completion as oai_Completion
from autogen.oai.openai_utils import get_key

from llm_gateway_client.configuration import Configuration as LLMGatewayConfiguration
from llm_gateway_client.api_client import ApiClient
from llm_gateway_client.api import DefaultApi
from llm_gateway_client.exceptions import ApiException


class Completion(oai_Completion):
    """A class for the LLM Gateway generation API.

    This is a version of the AutoGen `Completion` class, but wired to use the
    LLM Gateway APIs instead of the OpenAI one.
    """

    x_llm_provider = "OpenAI"
    x_org_id = "00DR0000000BLPeMAO"
    chat_session_id = None
    llm_gateway_config = None

    try:
        dotvals = dotenv_values()
        llm_gateway_config = LLMGatewayConfiguration(
            host=dotvals["LLM_GATEWAY_HOST"],
            access_token=dotvals["LLM_GATEWAY_TOKEN"],
            ssl_ca_cert=dotvals["SSL_CA_CERT"],
        )
    except Exception as e:
        raise e

    @classmethod
    def _get_response(cls, config: Dict, raise_on_ratelimit_or_timeout: bool = False, use_cache: bool = True):
        """Get the response from the LLM Gateway generation API call.

        Try the cache first. If not found, call the LLM Gateway API. If the API
        call fails, retry after `retry_wait_time`.
        """
        config = config.copy()
        key = get_key(config)
        if use_cache:
            response = cls._cache.get(key, None)
            if response is not None and (response != -1 or not raise_on_ratelimit_or_timeout):
                cls._book_keeping(config, response)
                return response
        model_id = config["model"].replace("gpt-35-turbo", "gpt-3.5-turbo")  # Normalize Azure ID
        use_chat_completion = model_id in cls.chat_models or issubclass(cls, ChatCompletion)
        # start_time = time()
        # request_timeout = cls.request_timeout
        # max_retry_period = config.pop("max_retry_period", cls.max_retry_period)
        # retry_wait_time = config.pop("retry_wait_time", cls.retry_wait_time)
        while True:
            try:
                event_loop = asyncio.get_event_loop()
                response = event_loop.run_until_complete(cls._complete_request(config, False))
                result = as_openai_response(response)
            except Exception as e:
                raise e
            else:
                if use_cache:
                    cls._cache.set(key, result)
                cls._book_keeping(config, result)
                return result

    @classmethod
    async def _create_chat_session(cls):
        async with ApiClient(configuration=cls.llm_gateway_config) as client:
            client.default_headers["X-Org-Id"] = cls.x_org_id
            default_api = DefaultApi(client)

            try:
                response = await default_api.create_chat_session(
                    x_llm_provider=cls.x_llm_provider, create_chat_session_request=None
                )
                cls.chat_session_id = response.session_details.session_id
            except Exception as e:
                raise e

    @classmethod
    async def _delete_chat_session(cls):
        if cls.chat_session_id is not None:
            async with ApiClient(configuration=cls.llm_gateway_config) as client:
                client.default_headers["X-Org-Id"] = cls.x_org_id
                default_api = DefaultApi(client)

                try:
                    _ = await default_api.delete_chat_session(str(cls.chat_session_id))
                    cls.chat_session_id = None
                except Exception as e:
                    raise e

    @classmethod
    async def _complete_request(cls, config: Dict, use_chat_completion: bool):
        if use_chat_completion:
            # check for chat session ID and use chat_generations endpoint
            raise NotImplementedError("chat completion has not been implemented")
        else:
            async with ApiClient(configuration=cls.llm_gateway_config) as client:
                client.default_headers["X-Org-Id"] = cls.x_org_id

                default_api = DefaultApi(client)
                gen_request = as_generation_request(config)
                try:
                    response = await default_api.generations(cls.x_llm_provider, gen_request, async_req=False)
                except ApiException as e:
                    raise e

                return response


class ChatCompletion(Completion):
    """A class for the LLM Gateway chat completion API."""
