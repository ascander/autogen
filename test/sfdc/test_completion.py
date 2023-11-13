from typing import Dict
import pytest
from autogen.sfdc.completion import ChatCompletion, Completion


def test_set_cache():
    seed = 666
    path = ".cache"
    Completion.set_cache(seed, path)

    assert Completion.seed == seed
    assert Completion.cache_path == f"{path}/{seed}"


def test_book_keeping():
    config = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    }
    response = Completion.create(**config)

    assert response.get("usage") is not None  # type:ignore
    assert response.get("usage", {}).get("total_tokens", 0) > 0  # type:ignore
    assert response.get("cost", 0.0) > 0.0  # type:ignore


def test_cost():
    response = {"model": "gpt-3.5-turbo", "usage": {"completion_tokens": 9, "prompt_tokens": 31, "total_tokens": 40}}

    assert Completion.cost(response) > 0


def test_instantiate():
    template = "Tell me a joke about {subject}"
    context = {"subject": "turnips"}

    assert "turnips" in Completion.instantiate(template, context, True)  # type:ignore
    assert "turnips" not in Completion.instantiate(template, context, False)  # type:ignore


def test_construct_params_no_prompt():
    with pytest.raises(ValueError):
        Completion._construct_params(context={}, config={"model": "gpt-4"})


def test_construct_params_templates():
    context = {"subject": "turnips"}
    config = {"model": "text-davinci-003", "prompt": "Tell me a joke about {subject}"}
    params = Completion._construct_params(context, config, allow_format_str_template=True)
    alt_params = Completion._construct_params(context, config, allow_format_str_template=False)

    assert params.get("prompt", "") == "Tell me a joke about turnips"
    assert alt_params.get("prompt", "") == config.get("prompt")


def test_construct_params():
    chatty_model = "gpt-3.5-turbo"
    no_chat_model = "text-davinci-003"
    prompt = "Hello!"

    completion_params = Completion._construct_params(context=None, config={"model": no_chat_model}, prompt=prompt)
    chatting_params = Completion._construct_params(context=None, config={"model": chatty_model}, prompt=prompt)

    assert completion_params["prompt"] == prompt
    assert chatting_params.get("messages") is not None
    assert chatting_params.get("messages", [])[0] == {"role": "user", "content": prompt}


def test_extract_text_completion():
    response = {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "gpt-3.5-turbo",
        "choices": [{"text": "This is indeed a test", "index": 0, "logprobs": None, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }

    assert Completion.extract_text(response) == ["This is indeed a test"]


def test_extract_text_chat_completion():
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0613",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there, how may I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }

    assert Completion.extract_text(response) == ["Hello there, how may I assist you today?"]


def test_extract_text_or_function_call():
    response = {
        "choices": [
            {
                "finish_reason": "function_call",
                "index": 0,
                "message": {
                    "content": None,
                    "function_call": {"name": "get_current_weather", "arguments": '{\n  "location": "Boston, MA"\n}'},
                    "role": "assistant",
                },
            }
        ],
        "created": 1694028367,
        "model": "gpt-3.5-turbo-0613",
        "system_fingerprint": "fp_44709d6fcb",
        "object": "chat.completion",
        "usage": {"completion_tokens": 18, "prompt_tokens": 82, "total_tokens": 100},
    }
    expected = [choice["message"] for choice in response["choices"]]

    assert Completion.extract_text_or_function_call(response) == expected


def test_logged_history():
    config = {"model": "text-davinci-003", "prompt": "Say this is a test."}
    Completion.start_logging()
    _ = Completion.create(**config)

    assert Completion.logged_history is not {}


def test_filter_func():
    model_list = ["text-davinci-002", "text-davinci-003", "gpt-3.5-turbo"]
    config_list = [{"model": model} for model in model_list]

    def yes_or_no_filter(context, response, **_):
        return context.get("yes_or_no_choice", False) is False or any(
            text in ["Yes.", "No."] for text in Completion.extract_text(response)
        )

    response = Completion.create(
        context={"yes_or_no_choice": True},
        config_list=config_list,
        prompt="Is 37 a prime number? Please answer 'Yes.' or 'No.'",
        filter_func=yes_or_no_filter,  # type:ignore
    )
    assert Completion.extract_text(response)[0] in ["Yes.", "No."]


def test_multi_model():
    models = ["gpt-4", "gpt-3.5-turbo"]
    config_list = [{"model": model} for model in models]

    response = Completion.create(config_list=config_list, prompt="Hi")

    assert response is not -1
