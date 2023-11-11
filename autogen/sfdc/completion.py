import asyncio
import diskcache
import logging
import numpy as np
import sys

from dotenv import dotenv_values
from time import sleep
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
            raise GATEWAY_ERROR  # type:ignore

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
                    )  # type:ignore
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
        params = cls._construct_params(
            context, config, allow_format_str_template=allow_format_str_template
        )  # type:ignore
        if not use_cache:
            logger.debug("skipping cache")
            return cls._get_response(
                params, raise_on_ratelimit_or_timeout=raise_on_ratelimit_or_timeout, use_cache=False
            )  # type:ignore
        seed = cls.seed
        if "seed" in params:
            cls.set_cache(params.pop("seed"))
        with diskcache.Cache(cls.cache_path) as cls._cache:
            cls.set_cache(seed)
            return cls._get_response(params, raise_on_ratelimit_or_timeout=raise_on_ratelimit_or_timeout)  # type:ignore

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
                    logger.debug("using chat completion")
                    # Create chat session if needed and call the chat completion endpoint
                    if cls.chat_session_id is None:
                        logger.debug("initializing chat session")
                        await cls._create_chat_session()
                    request = as_chat_generation_request(config)
                    logger.debug(f"calling chat completion for session {cls.chat_session_id}")
                    response = await api.chat_generations(str(cls.chat_session_id), request)
                    result = convert_chat_generation_response(response)
                else:
                    logger.debug("using completion")
                    # Call the completion endpoint
                    request = as_generation_request(config)
                    response = await api.generations(cls.x_llm_provider, request)
                    result = convert_generation_response(response)
            except ApiException as e:
                logger.warn(f"Caught exception: ({e.status}) {e.reason}")
                raise e

            return result

    @classmethod
    def _eval(cls, config: dict, prune=True, eval_only=False):
        """Evaluate the given config as the hyperparameter setting for the openai api call.

        Args:
            config (dict): Hyperparameter setting for the openai api call.
            prune (bool, optional): Whether to enable pruning. Defaults to True.
            eval_only (bool, optional): Whether to evaluate only
              (ignore the inference budget and do not rasie error when a request fails).
              Defaults to False.

        Returns:
            dict: Evaluation results.
        """
        cost = 0
        data = cls.data
        params = cls._get_params_for_create(config)
        model = params["model"]
        data_length = len(data)
        price = cls.price1K.get(model)
        price_input, price_output = price if isinstance(price, tuple) else (price, price)
        inference_budget = getattr(cls, "inference_budget", None)
        prune_hp = getattr(cls, "_prune_hp", "n")
        metric = cls._metric
        config_n = params.get(prune_hp, 1)  # default value in OpenAI is 1
        max_tokens = params.get(
            "max_tokens", np.inf if model in cls.chat_models or issubclass(cls, ChatCompletion) else 16
        )
        target_output_tokens = None
        if not cls.avg_input_tokens:
            input_tokens = [None] * data_length
        prune = prune and inference_budget and not eval_only
        if prune:
            region_key = cls._get_region_key(config)
            max_valid_n = cls._get_max_valid_n(region_key, max_tokens)
            if cls.avg_input_tokens:
                target_output_tokens = (
                    inference_budget * 1000 - cls.avg_input_tokens * price_input
                ) / price_output  # type:ignore
                # max_tokens bounds the maximum tokens
                # so using it we can calculate a valid n according to the avg # input tokens
                max_valid_n = max(
                    max_valid_n,
                    int(target_output_tokens // max_tokens),
                )
            if config_n <= max_valid_n:
                start_n = config_n
            else:
                min_invalid_n = cls._get_min_invalid_n(region_key, max_tokens)
                if min_invalid_n is not None and config_n >= min_invalid_n:
                    # prune this config
                    return {
                        "inference_cost": np.inf,
                        metric: np.inf if cls._mode == "min" else -np.inf,
                        "cost": cost,
                    }
                start_n = max_valid_n + 1
        else:
            start_n = config_n
            region_key = None
        num_completions, previous_num_completions = start_n, 0
        n_tokens_list, result, responses_list = [], {}, []
        while True:  # n <= config_n
            params[prune_hp] = num_completions - previous_num_completions
            data_limit = 1 if prune else data_length
            prev_data_limit = 0
            data_early_stop = False  # whether data early stop happens for this n
            while True:  # data_limit <= data_length
                # limit the number of data points to avoid rate limit
                for i in range(prev_data_limit, data_limit):
                    logger.debug(f"num_completions={num_completions}, data instance={i}")
                    data_i = data[i]
                    response = cls.create(data_i, raise_on_ratelimit_or_timeout=eval_only, **params)
                    if response == -1:  # rate limit/timeout error, treat as invalid
                        cls._update_invalid_n(prune, region_key, max_tokens, num_completions)
                        result[metric] = 0
                        result["cost"] = cost
                        return result
                    # evaluate the quality of the responses
                    responses = cls.extract_text_or_function_call(response)
                    usage = response["usage"]
                    n_input_tokens = usage["prompt_tokens"]
                    n_output_tokens = usage.get("completion_tokens", 0)
                    if not cls.avg_input_tokens and not input_tokens[i]:
                        # store the # input tokens
                        input_tokens[i] = n_input_tokens
                    query_cost = response["cost"]
                    cls._total_cost += query_cost
                    cost += query_cost
                    if cls.optimization_budget and cls._total_cost >= cls.optimization_budget and not eval_only:
                        # limit the total tuning cost
                        return {
                            metric: 0,
                            "total_cost": cls._total_cost,
                            "cost": cost,
                        }
                    if previous_num_completions:
                        n_tokens_list[i] += n_output_tokens
                        responses_list[i].extend(responses)
                        # Assumption 1: assuming requesting n1, n2 responses separatively then combining them
                        # is the same as requesting (n1+n2) responses together
                    else:
                        n_tokens_list.append(n_output_tokens)
                        responses_list.append(responses)
                avg_n_tokens = np.mean(n_tokens_list[:data_limit])
                rho = (
                    (1 - data_limit / data_length) * (1 + 1 / data_limit)
                    if data_limit << 1 > data_length
                    else (1 - (data_limit - 1) / data_length)
                )
                # Hoeffding-Serfling bound
                ratio = 0.1 * np.sqrt(rho / data_limit)
                if target_output_tokens and avg_n_tokens > target_output_tokens * (1 + ratio) and not eval_only:
                    cls._update_invalid_n(prune, region_key, max_tokens, num_completions)
                    result[metric] = 0
                    result["total_cost"] = cls._total_cost
                    result["cost"] = cost
                    return result
                if (
                    prune
                    and target_output_tokens
                    and avg_n_tokens <= target_output_tokens * (1 - ratio)
                    and (num_completions < config_n or num_completions == config_n and data_limit == data_length)
                ):
                    # update valid n
                    cls._max_valid_n_per_max_tokens[region_key] = valid_n = cls._max_valid_n_per_max_tokens.get(
                        region_key, {}
                    )
                    valid_n[max_tokens] = max(num_completions, valid_n.get(max_tokens, 0))
                    if num_completions < config_n:
                        # valid already, skip the rest of the data
                        data_limit = data_length
                        data_early_stop = True
                        break
                prev_data_limit = data_limit
                if data_limit < data_length:
                    data_limit = min(data_limit << 1, data_length)
                else:
                    break
            # use exponential search to increase n
            if num_completions == config_n:
                for i in range(data_limit):
                    data_i = data[i]
                    responses = responses_list[i]
                    metrics = cls._eval_func(responses, **data_i)
                    if result:
                        for key, value in metrics.items():
                            if isinstance(value, (float, int)):
                                result[key] += value
                    else:
                        result = metrics
                for key in result.keys():
                    if isinstance(result[key], (float, int)):
                        result[key] /= data_limit
                result["total_cost"] = cls._total_cost
                result["cost"] = cost
                if not cls.avg_input_tokens:
                    cls.avg_input_tokens = np.mean(input_tokens)
                    if prune:
                        target_output_tokens = (
                            inference_budget * 1000 - cls.avg_input_tokens * price_input
                        ) / price_output
                result["inference_cost"] = (avg_n_tokens * price_output + cls.avg_input_tokens * price_input) / 1000
                break
            else:
                if data_early_stop:
                    previous_num_completions = 0
                    n_tokens_list.clear()
                    responses_list.clear()
                else:
                    previous_num_completions = num_completions
                num_completions = min(num_completions << 1, config_n)
        return result

    @classmethod
    def _construct_params(cls, context, config, prompt=None, messages=None, allow_format_str_template=False):
        params = config.copy()
        model = config["model"]
        prompt = config.get("prompt") if prompt is None else prompt
        messages = config.get("messages") if messages is None else messages
        # either "prompt" should be in config (for being compatible with non-chat models)
        # or "messages" should be in config (for tuning chat models only)
        if prompt is None and (model in cls.chat_models or issubclass(cls, ChatCompletion)):
            if messages is None:
                raise ValueError("Either prompt or messages should be in config for chat models.")
        if prompt is None:
            params["messages"] = (
                [
                    {
                        **m,
                        "content": cls.instantiate(m["content"], context, allow_format_str_template),
                    }
                    if m.get("content")
                    else m
                    for m in messages
                ]
                if context
                else messages
            )
        elif model in cls.chat_models or issubclass(cls, ChatCompletion):
            # convert prompt to messages
            params["messages"] = [
                {
                    "role": "user",
                    "content": cls.instantiate(prompt, context, allow_format_str_template),
                },
            ]
            params.pop("prompt", None)
        else:
            params["prompt"] = cls.instantiate(prompt, context, allow_format_str_template)
        return params


class ChatCompletion(Completion):
    """A class for the LLM Gateway chat completion API."""
