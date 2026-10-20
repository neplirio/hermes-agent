# Native search chain prototype

## Configuration and behavior

```yaml
web:
  search_backends: [native, firecrawl, exa]
  extract_backend: firecrawl
```

This is opt-in. The presence of `search_backends` replaces search selection;
it does not append a free/managed rescue tier. The ordered list is pinned at
agent construction. Every model request resolves it again against the effective
provider, base endpoint, actual transport class and wire model (including a
top-level `request_overrides.model`). Model/provider switches need no config edit.
Config edits take effect in a new session.

`native` means the current inference executes a hosted tool. It never starts a
second inference to implement a client function. Unsupported/unknown native
routes and unavailable client providers are skipped, with INFO diagnostics and
reasons on the immutable selection. An exhausted chain retains the logical
function schema but returns a failure with diagnostics if the model calls it.
Direct programmatic `web_search` without a model request skips `native`.

Lists must be nonempty, contain unique nonempty names, and name registered search
providers (or `native`). Unknown/disabled plugins fail before the model API call,
even if an earlier candidate could serve. Shape errors fail at construction.
Use `native` instead of the legacy `openai-native` marker inside a chain.

## Capability boundary

LLM `ProviderProfile.native_search_rules` owns the allowlist. Every rule requires
an exact HTTPS endpoint/path, exact model ID, `codex_responses` API mode and the
registered Responses transport implementation. Only a trailing slash is ignored;
provider aliases resolve through the existing provider registry. Names containing
`gpt` or `grok` do not confer support, and proxies inherit none.

The initial narrow allowlist uses the repository's existing native transport
fixtures and xAI regression evidence:

| Provider | Base endpoint | Exact models |
|---|---|---|
| `openai-codex` | `https://chatgpt.com/backend-api/codex` | `gpt-5-codex`, `gpt-5.4` |
| `xai` | `https://api.x.ai/v1` | `grok-composer-2.5-fast` |

This prototype has **no live API verification**. The tests exercise real request
construction and dispatch with fake provider results. New model IDs need a
provider-owned rule backed by route verification; they are not automatically
enabled. Direct OpenAI API, xAI OAuth-specific routes, Anthropic, Gemini and
arbitrary compatible proxies remain unknown/unsupported for generic native search.

## Lifecycle and implementation

`agent/web_search_selection.py` creates a frozen `SearchSelection` on each request.
It holds the selected provider instance or native tool type and diagnostic tuple.
The agent keeps the latest request selection; there is no process-global selected
backend, model or chain cursor. The existing session turn lease serializes each
agent's request/tool round. The sequential and concurrent tool entry points bind
that same selection in a ContextVar before spawning workers. Existing
`propagate_context_to_thread` carries it with profile/secret scope; `finally`
restores the prior context. An already-started worker keeps its old selection
when a later request changes the agent's latest selection.

The logical `web_search` schema and profile-level availability remain stable
across model switches. The Responses transport performs native substitution only
when the request already contains that function. Unsupported xAI routes retain
the existing wire alias and reverse mapping. Disabling tools grants no native
search. Chain configuration cannot be combined with `request_overrides.tools`,
`extra_body.tools` or `extra_body.model`, which could bypass that declaration;
top-level `request_overrides.model` is supported.

Client dispatch uses the pinned provider rather than re-resolving legacy config.
Bundled client readiness excludes anonymous fallback and Free tier pins. Firecrawl
requires a direct key or self-hosted URL and its chain search ignores legacy
managed selection. A provider readiness exception skips that candidate;
`UnscopedSecretError` propagates instead of borrowing launch-profile credentials.

For explicit chains only, client search bypasses the old process-wide memo to
avoid cross-profile/config result reuse and cached legacy rescue semantics. This
costs repeat client calls but avoids a cache refactor. Hosted search and the
independent extract cache are unchanged. Third-party providers must honor
`is_available_for_search_chain()` and avoid hidden search rescue themselves.
Firecrawl chain searches also construct a local SDK client independently of the
legacy client cache. This bypass is limited to explicit-chain search. The legacy
Firecrawl cache now publishes a single config/client tuple and returns the local
client after construction, preventing concurrent profiles from mixing a config
with another client's credentials. Resetting `tools.web_tools._firecrawl_client`
to `None` still forces reconstruction; unchanged configurations still reuse it.

## Compatibility and limits

Without the new key, `search_backend > backend > autodetect`, explicit
`openai-native`/`xai`, legacy caching, managed fallback and keyless rescue keep
their existing behavior. Extract selection, execution, free tiers and rescue
are independent, including during a native search turn.

V1 fallback is **selection-time only**. HTTP errors, rate limits, credential
revocation, network failures and hosted tool failures do not advance the chain.
The selected client's error is returned; a hosted failure follows existing model
error handling. No search-specific whole-model retry is added. Existing SDK,
transport and agent retry policies still apply and are not disabled here.

## Review and validation

The prototype tests live in `tests/agent/test_native_search_chain.py`. They include
real AIAgent request construction, actual `switch_model`, sequential/concurrent
tool dispatch, two temporary profile config files, interleaved scopes, disabled
tools, extract, old selections/rescue and hosted-error request counting. They are
collectable by pytest and runnable with unittest when offline.
The unittest entry point also loads the three actual-import, fake-client cache
regressions in `tests/plugins/web/test_firecrawl_client_cache.py`: concurrent
construction, replacement during lookup, and visible cache reset. These use
events to force interleavings and assert returned-client behavior, without
requiring separate cache slots or a fixed publication order.

Use `scripts/run_tests.sh` for normal validation, including existing
`tests/agent/transports/test_codex_transport.py`,
`tests/agent/test_run_agent_codex_responses.py`,
`tests/plugins/web/test_web_search_provider_plugins.py`,
`tests/tools/test_web_keyless_rescue.py` and `tests/tools/test_web_result_cache.py`.
Local validation used the canonical `scripts/run_tests.sh` with an isolated copy
of the runtime plus pinned pytest 9.1.1 and pytest-asyncio 1.3.0. The eight focused
files (the seven listed above plus `tests/tools/test_web_tools_config.py`, counting
the two new test files) passed: **370 tests, zero failures**. The standalone
unittest run also passed all 29 new behaviors. These use fake provider clients;
no live search API calls or full-repository regression run were performed.

A fresh read-only Codex review found no blocking issues within the prototype
scope. Follow-ups include actual disabled-plugin discovery coverage and a
permanent same-key/different-endpoint cache regression. The fork branch is for
manual review, not a production deployment.
