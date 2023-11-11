import autogen
import json

from autogen.sfdc.completion import Completion


def yes_or_no_filter(context, response, **_):
    return context.get("yes_or_no_choice", False) is False or any(
        text in ["Yes.", "No."] for text in Completion.extract_text(response)
    )

def valid_json_filter(response, **_):
    for text in autogen.Completion.extract_text(response):
        try:
            json.loads(text)
            return True
        except ValueError:
            pass
    return False

def test_filter():
    config_list = [
            { "model": "gpt-4" },
            { "model": "gpt-3.5-turbo" },
            { "model": "text-davinci-003"}
    ]
    response = Completion.create(
            context={"yes_or_no_choice": True},
            config_list=config_list,
            prompt="Is 37 a prime number? Please answer 'Yes.' or 'No.'",
            filter_func=yes_or_no_filter
    )
    assert (
        Completion.extract_text(response)[0] in ["Yes.", "No."]
        or not response["pass_filter"]
        and response["config_id"] == 2
    )
    response = Completion.create(
        context={"yes_or_no_choice": False},
        config_list=config_list,
        prompt="Is 37 a prime number?",
        filter_func=yes_or_no_filter,
    )
    assert response["model"] == "gpt-4-0613"
    response = Completion.create(
        config_list=config_list,
        prompt="How to construct a json request to Bing API to search for 'latest AI news'? Return the JSON request.",
        filter_func=valid_json_filter,
    )
    assert response["config_id"] == 2 or response["pass_filter"], "the response must pass filter unless all fail"
    assert not response["pass_filter"] or json.loads(Completion.extract_text(response)[0])

