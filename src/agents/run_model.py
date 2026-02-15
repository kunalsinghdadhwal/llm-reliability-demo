"""run_model -- run test prompts against the target LLM."""

import os

from src.wrappers.bedrock import call_llm


def _build_system_prompt(use_case: str) -> str:
    """Build a system prompt grounding the model in its use case."""
    return (
        f"You are a {use_case}. "
        "Answer questions based ONLY on the company's official policies. "
        "If you are not sure about a specific policy detail, say you would need "
        "to check or refer the customer to the official documentation. "
        "Do not make generic statements about what other companies typically do. "
        "Do not discuss your own capabilities or limitations as an AI."
    )


def call_target_llm(prompt: str, model_config: dict, use_case: str = "") -> str:
    """Route a prompt to the correct provider."""
    provider = model_config["provider"]
    system = _build_system_prompt(use_case) if use_case else ""

    if provider == "bedrock":
        return call_llm(prompt, system=system)
    elif provider == "ollama":
        from ollama import Client as OllamaClient

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        client = OllamaClient(host=host)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat(
            model=model_config["model_id"],
            messages=messages,
        )
        return response.message.content or ""
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_model(state: dict) -> None:
    """Run each prompt against the target LLM and collect responses."""
    prompts = state["prompts"]
    model_config = state["config"]["model"]

    use_case = state["config"].get("use_case", "")

    responses = []
    for prompt in prompts:
        result = call_target_llm(prompt, model_config, use_case=use_case)
        responses.append({"prompt": prompt, "response": result})

    state["responses"] = responses
