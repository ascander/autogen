import asyncio
import logging
import sys
import diskcache

from dotenv import dotenv_values
from time import sleep, time
from typing import Callable, Dict, List, Optional

from .utils import (
    as_chat_generation_request,
    as_generation_request,
    convert_chat_generation_response,
    convert_generation_response,
)

from autogen.oai.completion import Completion as oai_Completion
from autogen.oai.openai_utils import get_key

from llm_gateway_client.configuration import Configuration as LLMGatewayConfiguration
from llm_gateway_client.api_client import ApiClient
from llm_gateway_client.api.default_api import DefaultApi
from llm_gateway_client.exceptions import ApiException, ServiceException


# Initialize and configure logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter("[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _ch = logging.StreamHandler(stream=sys.stdout)
    _ch.setFormatter(formatter)
    logger.addHandler(_ch)

# Initialize LLM Gateway config error
GATEWAY_ERROR = None


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
        GATEWAY_ERROR = RuntimeError(
            f"Got exception {e} initializing the LLM Gateway configuration. Please ensure there is a suitable '.env' file with the LLM Gateway host, access token, and path to SSL cert available."
        )

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
                logger.info(f"using cached response for key {key}")
                cls._book_keeping(config, response)
                return response
        # Normalize the model ID since Azure does it differently
        model_id = config.get("model", "").replace("gpt-35-turbo", "gpt-3.5-turbo")
        use_chat_completion = model_id in cls.chat_models or issubclass(cls, ChatCompletion)
        retry_wait_time = config.pop("retry_wait_time", cls.retry_wait_time)
        while True:
            try:
                result = asyncio.run(cls._complete_request(config, use_chat_completion))
            except ServiceException as e:
                # HTTP status codes 5XX - may be transient
                logger.info(f"got service exception ({e.status}) {e.reason} - retrying in {retry_wait_time}s")
                sleep(retry_wait_time)
            except ApiException as e:
                # HTTP status codes 4XX - retrying is probably a waste of time
                logger.warn(f"got API exception ({e.status}) {e.reason} - returning -1")
                response = -1
                if use_cache:
                    logger.debug(f"caching empty response for key {key}")
                    cls._cache.set(key, response)
                    return response
            else:
                if use_cache:
                    logger.debug(f"caching response for key {key}")
                    cls._cache.set(key, result)
                cls._book_keeping(config, result)
                return result

    @classmethod
    def create(
        cls,
        context: Optional[Dict] = None,
        use_cache: Optional[bool] = True,
        config_list: Optional[List[Dict]] = None,
        filter_func: Optional[Callable[[Dict, Dict, Dict], bool]] = None,
        raise_on_ratelimit_or_timeout: Optional[bool] = True,
        allow_format_str_template: Optional[bool] = False,
        **config,
    ):
        """Make a completion for the given context.

        Args:
            context (Dict, Optional): The context to instantiate the prompt.
                The context holds keys/values to be instantiated into the prompt or
                filter function for the LLM call.
            use_cache (bool, Optional): Whether to use cached responses.
            config_list (List, Optional): List of configurations for the
                completion to try. The first that does not raise an error will be
                used. Only the differences from the default config need to be
                provided.
            filter_func (Callable, Optional): A function that takes in the
                context, the config, and the response and returns a boolean to
                indicate whether the response is valid. For example:

                ```python
                def yes_or_no_filter(context: Dict, config: Dict, response: Dict) -> bool:
                    return context.get("yes_or_no_choice", False) is False or any(
                        text in ["Yes.", "No."] for text in sfdc.Completion.extract_text(response)
                    )
                ```
            raise_on_ratelimit_or_timeout (bool, Optional): Whether to raise an
                error on a rate limit error or timeout when all configs fail. Note:
                this is not yet implemented for the LLM Gateway, as it does not
                return specific exception types for these cases.
            allow_format_str_template (bool, Optional): Whether to allow format
                string template in the config.
            **config: Configuration for the API call. This is used as
                parameters for calling the LLM Gateway. The "prompt" or "messages"
                parameter must be present and contain a template (str or Callable)
                which will be instantiated with the context. Besides the parameters
                for the LLM Gateway call, it can also contain the following
                entries:
                    - `max_retry_period` (int): the total time (in seconds) allowed for retrying failed requests.
                    - `retry_wait_time` (int): the time interval to wait (in seconds) before retrying.
                    - `seed` (int): a seed for the cache. This is useful for "controlled randomness".

        Returns:
            Responses from the LLM Gateway, cast as the equivalent OpenAI
            return objects, with the following additional fields:
                - `cost`: the estimated cost of the call, based on token usage and pricing
            When `config_list` is provided, the response will contain a few more fields:
                - `config_id`: the index of the config in the config list that was used.
                - `pass_filter`: whether the response passes the filter function. None if no filter is provided.
        """
        if GATEWAY_ERROR:
            raise GATEWAY_ERROR

        # Warn if a config list was provided but is empty
        if type(config_list) is list and len(config_list) == 0:
            logger.warning("Completion was provded with a config list, but the list was empty.")

        if config_list:
            last = len(config_list) - 1
            cost = 0
            for i, each_config in enumerate(config_list):
                base_config = config.copy()
                base_config["allow_format_str_template"] = allow_format_str_template
                base_config.update(each_config)
                if i < last and filter_func is None and "max_retry_period" not in base_config:
                    # max_retry_period = 0 to avoid retrying when no filter is given
                    base_config["max_retry_period"] = 0
                try:
                    response = cls.create(
                        context,
                        use_cache,
                        raise_on_ratelimit_or_timeout=i < last or raise_on_ratelimit_or_timeout,
                        **base_config,
                    )
                    if response == -1:
                        return response
                    pass_filter = filter_func is None or filter_func(
                        context=context, base_config=config, response=response
                    )
                    if pass_filter or i == last:
                        response["cost"] = cost + response["cost"]
                        response["config_id"] = i
                        response["pass_filter"] = pass_filter
                        return response
                    cost += response["cost"]
                except ApiException as e:
                    logger.debug(f"completion failed with ({e.status}) {e.reason} with config {i}", exc_info=True)
                    if i == last:
                        raise
        params = cls._construct_params(context, config, allow_format_str_template=allow_format_str_template)
        if not use_cache:
            logger.debug("skipping cache")
            return cls._get_response(
                params, raise_on_ratelimit_or_timeout=raise_on_ratelimit_or_timeout, use_cache=False
            )
        seed = cls.seed
        if "seed" in params:
            cls.set_cache(params.pop("seed"))
        with diskcache.Cache(cls.cache_path) as cls._cache:
            cls.set_cache(seed)
            return cls._get_response(params, raise_on_ratelimit_or_timeout=raise_on_ratelimit_or_timeout)

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
    async def _complete_request(cls, config: Dict, use_chat_completion: bool = False):
        async with ApiClient(configuration=cls.llm_gateway_config) as client:
            client.default_headers["X-Org-Id"] = cls.x_org_id
            api = DefaultApi(client)
            try:
                if use_chat_completion:
                    # Create chat session if needed and call the chat completion endpoint
                    if cls.chat_session_id is None:
                        await cls._create_chat_session()
                    request = as_chat_generation_request(config)
                    response = await api.chat_generations(str(cls.chat_session_id), request)
                    result = convert_chat_generation_response(response)
                else:
                    # Call the completion endpoint
                    request = as_generation_request(config)
                    response = await api.generations(cls.x_llm_provider, request)
                    result = convert_generation_response(response)
            except ApiException as e:
                logger.warn(f"Caught exception: ({e.status}) {e.reason}")
                raise e

            return result


class ChatCompletion(Completion):
    """A class for the LLM Gateway chat completion API."""
