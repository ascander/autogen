from typing import Dict
from llm_gateway_client.models.generation_request import GenerationRequest
from llm_gateway_client.models.generation_response import GenerationResponse
from openai.openai_object import OpenAIObject


def as_generation_request(config: Dict) -> GenerationRequest:
    return GenerationRequest.from_dict(config)


def as_openai_response(gen_response: GenerationResponse) -> OpenAIObject:
    gen_dict = gen_response.to_dict()
    oai_dict = {
        "id": gen_dict["id"],
        "object": gen_dict["parameters"]["object"],
        "created": gen_dict["parameters"]["created"],
        "model": gen_dict["parameters"]["model"],
        "usage": gen_dict["parameters"]["usage"],
        "choices": [
            {
                "index": msg["parameters"]["index"],
                "text": msg["text"],
                "finish_reason": msg["parameters"]["finish_reason"],
                "logprobs": msg["parameters"]["logprobs"],
            }
            for msg in gen_dict["generations"]
        ],
    }

    return OpenAIObject.construct_from(oai_dict)
