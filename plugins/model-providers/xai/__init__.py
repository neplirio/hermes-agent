"""xAI (Grok) provider profile."""

from hermes_cli.version_info import get_version_info
from providers import register_provider
from providers.base import ProviderProfile
from providers.native_search import NativeSearchRule

xai = ProviderProfile(
    name="xai", aliases=("grok", "x-ai", "x.ai"), api_mode="codex_responses", env_vars=("XAI_API_KEY",),
    base_url="https://api.x.ai/v1", auth_type="api_key",
    native_search_rules=(NativeSearchRule(
        endpoint="https://api.x.ai/v1", models=("grok-composer-2.5-fast",),
    ),),
    default_headers={"User-Agent": f"Hermes-Agent/{get_version_info().base_version}"},
)

register_provider(xai)
