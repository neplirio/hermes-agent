"""One immutable search selection per model request; no runtime chain retries."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import logging

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchSelection:
    backend: str | None
    native_tool: str | None = None
    provider: WebSearchProvider | None = None
    diagnostics: tuple[str, ...] = ()


_CURRENT_SEARCH: ContextVar[SearchSelection | None] = ContextVar("web_search_selection")


def current_search_selection() -> SearchSelection | None:
    return _CURRENT_SEARCH.get(None)


def search_selection_for_tool() -> SearchSelection | None:
    try:
        # Bound None means a legacy request: it must not adopt a chain that was
        # added to config after declaration. Only standalone tools read config.
        return _CURRENT_SEARCH.get()
    except LookupError:
        from hermes_cli.config import load_config_readonly
        chain = parse_search_chain(load_config_readonly().get("web") or {})
        return select_search(chain) if chain is not None else None


@contextmanager
def search_selection_scope(selection: SearchSelection | None):
    token = _CURRENT_SEARCH.set(selection)
    try:
        yield
    finally:
        _CURRENT_SEARCH.reset(token)


def bind_agent_search_selection(execute):
    """Bind before spawning tool workers; the existing thread bridge copies it.

    Timed-out workers retain the old immutable choice even if a later model
    request has replaced the agent's latest selection.
    """
    @wraps(execute)
    def bound(agent, *args, **kwargs):
        with search_selection_scope(getattr(agent, "_web_search_selection", None)):
            return execute(agent, *args, **kwargs)
    return bound


def parse_search_chain(web_config: dict) -> tuple[str, ...] | None:
    if "search_backends" not in web_config:
        return None
    raw = web_config["search_backends"]
    if not isinstance(raw, list) or not raw or any(not isinstance(n, str) or not n.strip() for n in raw):
        raise ValueError("web.search_backends must be a non-empty list of backend names")
    names = tuple(n.strip().lower() for n in raw)
    if len(set(names)) != len(names):
        raise ValueError("web.search_backends must not contain duplicate names")
    if "openai-native" in names:
        raise ValueError("Use native in web.search_backends; openai-native is a legacy single-backend setting")
    return names


def native_search_tool(agent, model: str) -> str | None:
    from providers import get_provider_profile

    profile = get_provider_profile(agent.provider)
    if profile is None:
        return None
    transport = agent._get_transport()
    transport_name = f"{type(transport).__module__}.{type(transport).__qualname__}"
    for rule in profile.native_search_rules:
        if rule.matches(endpoint=agent.base_url, model=model,
                        api_mode=agent.api_mode, transport=transport_name):
            return rule.tool_type
    return None


def _client_ready(provider) -> tuple[bool, str]:
    from agent.secret_scope import UnscopedSecretError
    try:
        ready = provider.supports_search() and provider.is_available_for_search_chain()
        return bool(ready), "search unsupported or credentials/endpoint unavailable"
    except UnscopedSecretError:
        raise
    except Exception as exc:
        logger.warning("Search provider availability probe failed", exc_info=True)
        return False, f"availability probe raised {type(exc).__name__}"


def select_search(chain: tuple[str, ...], *, agent=None, model: str = "", enabled: bool = True) -> SearchSelection:
    from agent.web_search_registry import list_providers
    from tools.web_tools import _ensure_web_plugins_loaded

    _ensure_web_plugins_loaded()
    providers = {p.name: p for p in list_providers()}
    unknown = set(chain) - {"native"} - providers.keys()
    if unknown:
        raise ValueError(f"Unknown web.search_backends: {', '.join(sorted(unknown))} (check disabled plugins)")
    if not enabled:
        return SearchSelection(None, diagnostics=("web_search is disabled for this request",))
    diagnostics = []
    for name in chain:
        if name == "native":
            tool = native_search_tool(agent, model) if agent is not None else None
            if tool and agent.api_key:
                return SearchSelection(name, native_tool=tool, diagnostics=tuple(diagnostics))
            reason = "unsupported or unknown provider/endpoint/transport/model, or missing inference credentials"
        else:
            provider = providers[name]
            ready, reason = _client_ready(provider)
            if ready:
                return SearchSelection(name, provider=provider, diagnostics=tuple(diagnostics))
        diagnostic = f"web.search_backends: skipping {name}: {reason}"
        diagnostics.append(diagnostic)
        logger.info(diagnostic)
    return SearchSelection(None, diagnostics=tuple(diagnostics))


def prepare_search_selection(agent, tools, request_overrides: dict) -> SearchSelection | None:
    # AIAgent pins configuration at construction. Small embedded callers using the
    # request helper directly can pin it on their first request instead.
    if not hasattr(agent, "_web_search_chain"):
        from hermes_cli.config import load_config_readonly
        agent._web_search_chain = parse_search_chain(load_config_readonly().get("web") or {})
    chain = agent._web_search_chain
    if chain is None:
        agent._web_search_selection = None
        return None
    extra_body = request_overrides.get("extra_body") or {}
    if "tools" in request_overrides or any(k in extra_body for k in ("tools", "model")):
        raise ValueError("web.search_backends cannot be combined with tools overrides or extra_body.model")
    model = request_overrides.get("model", agent.model)
    if agent.api_mode == "codex_responses":
        from agent.model_metadata import strip_codex_context_variant_suffix
        model = strip_codex_context_variant_suffix(model)
    selection = select_search(
        chain, agent=agent, model=model,
        enabled=any(t.get("function", {}).get("name") == "web_search" for t in (tools or [])),
    )
    agent._web_search_selection = selection
    return selection
