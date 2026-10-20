"""OpenAI Codex (Responses API) provider profile."""

from providers import register_provider
from providers.base import ProviderProfile
from providers.native_search import NativeSearchRule

openai_codex = ProviderProfile(
    name="openai-codex", aliases=("codex", "openai_codex"), api_mode="codex_responses",
    env_vars=(),  # OAuth external — no API key
    base_url="https://chatgpt.com/backend-api/codex", auth_type="oauth_external",
    native_search_rules=(NativeSearchRule(
        endpoint="https://chatgpt.com/backend-api/codex",
        models=("gpt-5-codex", "gpt-5.4"),
    ),),
)

register_provider(openai_codex)
