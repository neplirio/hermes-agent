"""Request construction and real web-tool dispatch contracts for the opt-in chain.

Uses unittest as well as pytest collection, so the prototype can be exercised
offline when the checkout's PM test dependencies cannot be downloaded.
"""

import copy
import asyncio
import contextvars
from contextlib import closing
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent
from tools.web_tools import WEB_EXTRACT_SCHEMA, WEB_SEARCH_SCHEMA
from hermes_cli.config import load_config_readonly as real_load_config_readonly, load_config as real_load_config


class NativeSearchChainTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict("os.environ", {"HERMES_HOME": str(self.home)}))
        self.enterContext(patch.object(Path, "home", return_value=self.home))
        self.config = {"web": {"search_backends": ["native", "firecrawl", "exa"]}}
        self.enterContext(patch("hermes_cli.config.load_config_readonly", side_effect=lambda: self.config))
        self.enterContext(patch("hermes_cli.config.load_config", side_effect=lambda: self.config))
        self.enterContext(patch("model_tools.get_tool_definitions", return_value=[]))
        self.enterContext(patch("model_tools.check_toolset_requirements", return_value={}))
        self.enterContext(patch("agent.agent_init._setup_logging"))

    def agent(self, model="gpt-5-codex", provider="openai-codex",
              endpoint="https://chatgpt.com/backend-api/codex", api_mode="codex_responses"):
        agent = AIAgent(
            model=model, provider=provider, base_url=endpoint, api_mode=api_mode,
            api_key="test-placeholder", quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
        agent.tools = [{"type": "function", "function": copy.deepcopy(schema)}
                       for schema in (WEB_SEARCH_SCHEMA, WEB_EXTRACT_SCHEMA)]
        return agent

    def test_verified_native_substitutes_only_search_in_real_request(self):
        agent = self.agent()
        logical = copy.deepcopy(agent.tools)
        body = agent._build_api_kwargs([{"role": "user", "content": "Search for Hermes"}])
        self.assertIn({"type": "web_search"}, body["tools"])
        self.assertFalse(any(t.get("name") == "web_search" for t in body["tools"]))
        self.assertTrue(any(t.get("name") == "web_extract" for t in body["tools"]))
        self.assertEqual(logical, agent.tools)

    def test_client_dispatch_keeps_request_selection_in_tool_worker(self):
        from agent.tool_executor import execute_tool_calls_sequential
        self.config["web"]["search_backend"] = "firecrawl"
        self.enterContext(patch("plugins.web.firecrawl.provider.FirecrawlWebSearchProvider.is_available", return_value=False))
        self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.is_available", return_value=True))
        search = self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.search", return_value={
            "success": True, "data": {"web": []}, "selected": "exa",
        }))
        wrong = self.enterContext(patch("plugins.web.firecrawl.provider.FirecrawlWebSearchProvider.search", return_value={
            "success": False, "error": "wrong provider",
        }))
        agent = self.agent(model="unknown-model")
        agent.valid_tool_names = {"web_search", "web_extract"}
        agent._flush_messages_to_session_db = lambda *a, **kw: True
        body = agent._build_api_kwargs([{"role": "user", "content": "search"}])
        self.assertNotIn({"type": "web_search"}, body["tools"])
        # Editing config after declaration must not change this request's dispatch.
        self.config["web"]["search_backends"] = ["firecrawl"]
        messages = []
        call = SimpleNamespace(id="call-search", function=SimpleNamespace(name="web_search", arguments='{"query":"selection"}'))
        execute_tool_calls_sequential(agent, SimpleNamespace(tool_calls=[call]), messages, "chain-test", finalize=False)
        self.assertIn('"selected"', messages[-1]["content"])
        content = messages[-1]["content"]
        result, _ = json.JSONDecoder().raw_decode(content[content.index("{"):])
        self.assertEqual("exa", result.get("selected"), result)
        search.assert_called_once()
        wrong.assert_not_called()

    def test_effective_wire_model_controls_native_support(self):
        agent = self.agent()
        agent.request_overrides = {"model": "unverified-model"}
        body = agent._build_api_kwargs([{"role": "user", "content": "search"}])
        self.assertEqual("unverified-model", body["model"])
        self.assertNotIn({"type": "web_search"}, body["tools"])

    def test_explicit_chain_does_not_take_legacy_firecrawl_keyless_availability(self):
        self.config["web"]["search_backend"] = "firecrawl"
        self.enterContext(patch("plugins.web.firecrawl.provider._env", return_value=""))
        self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.is_available", return_value=True))
        agent = self.agent(model="unknown-model")
        agent._build_api_kwargs([{"role": "user", "content": "search"}])
        self.assertEqual("exa", agent._web_search_selection.backend)
        self.assertTrue(any("firecrawl" in d for d in agent._web_search_selection.diagnostics))

    def test_chain_firecrawl_executes_direct_route_despite_old_managed_selection(self):
        from tools.registry import registry
        from agent.web_search_selection import search_selection_scope
        self.config["web"]["search_backend"] = "nous"
        self.config["web"]["search_backends"] = ["firecrawl", "exa"]
        self.enterContext(patch("plugins.web.firecrawl.provider._env", side_effect=lambda k: "test-key" if k == "FIRECRAWL_API_KEY" else ""))
        client = SimpleNamespace(search=lambda **kw: {"web": []})
        factory = self.enterContext(patch("plugins.web.firecrawl.provider.Firecrawl", return_value=client))
        managed = self.enterContext(patch("plugins.web.firecrawl.provider._gateway.resolve_managed_tool_gateway", return_value=None))
        self.enterContext(patch("tools.web_tools._firecrawl_client", None))
        agent = self.agent(model="unknown-model")
        agent._build_api_kwargs([{"role": "user", "content": "search"}])
        with search_selection_scope(agent._web_search_selection):
            result = json.loads(registry.dispatch("web_search", {"query": "direct"}))
        self.assertTrue(result.get("success"), result)
        factory.assert_called_once_with(api_key="test-key")
        managed.assert_not_called()

    def test_unknown_routes_never_get_native_even_with_trusted_model_name(self):
        cases = [
            ("openai-codex", "https://proxy.example/v1", "gpt-5-codex", "codex_responses"),
            ("openai-codex", "https://chatgpt.com/backend-api/codex/other", "gpt-5-codex", "codex_responses"),
            ("openai-codex", "https://evil.chatgpt.com/backend-api/codex", "gpt-5-codex", "codex_responses"),
            ("openai-codex", "http://chatgpt.com/backend-api/codex", "gpt-5-codex", "codex_responses"),
            ("openai-codex", "https://chatgpt.com:8443/backend-api/codex", "gpt-5-codex", "codex_responses"),
            ("openai-codex", "https://api.openai.com/v1", "gpt-5-codex", "codex_responses"),
            ("xai", "https://proxy.example/v1", "grok-composer-2.5-fast", "codex_responses"),
            ("xai", "https://api.x.ai/v1", "unverified-grok", "codex_responses"),
            ("openai", "https://api.openai.com/v1", "gpt-5-codex", "chat_completions"),
        ]
        for provider, endpoint, model, mode in cases:
            with self.subTest(provider=provider, endpoint=endpoint, model=model):
                agent = self.agent(model, provider, endpoint, mode)
                body = agent._build_api_kwargs([{"role": "user", "content": "search"}])
                self.assertNotIn({"type": "web_search"}, body.get("tools", []))

    def test_sequential_model_switches_recompute_without_mutating_chain_or_schema(self):
        agent = self.agent()
        logical = copy.deepcopy(agent.tools)
        original = copy.deepcopy(self.config)
        for model, native in [("gpt-5-codex", True), ("unknown-model", False), ("gpt-5-codex", True)]:
            agent.model = model
            body = agent._build_api_kwargs([{"role": "user", "content": "search"}])
            self.assertEqual(native, {"type": "web_search"} in body["tools"])
            self.assertEqual(logical, agent.tools)
        self.assertEqual(original, self.config)

    def test_tools_disabled_never_grant_native_and_gate_is_model_independent(self):
        from tools.web_tools import check_web_api_key
        agent = self.agent()
        self.assertTrue(check_web_api_key())
        body = agent._build_api_kwargs([{"role": "user", "content": "hello"}], tools_for_api=[])
        self.assertFalse(body.get("tools"))
        self.assertIsNone(agent._web_search_selection.backend)
        agent.model = "unknown"
        self.assertTrue(check_web_api_key())

    def test_xai_native_and_unknown_model_alias_remain_request_local(self):
        from agent.tool_executor import execute_tool_calls_sequential
        self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.is_available", return_value=True))
        search = self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.search", return_value={
            "success": True, "selected": "xai-client-exa", "data": {"web": []},
        }))
        agent = self.agent("grok-composer-2.5-fast", "xai", "https://api.x.ai/v1")
        first = agent._build_api_kwargs([{"role": "user", "content": "search"}])
        self.assertIn({"type": "web_search"}, first["tools"])
        agent.model = "unverified-grok"
        second = agent._build_api_kwargs([{"role": "user", "content": "search"}])
        self.assertNotIn({"type": "web_search"}, second["tools"])
        self.assertTrue(any(t.get("name") == "hermes_web_search" for t in second["tools"]))
        response = SimpleNamespace(output=[SimpleNamespace(
            type="function_call", id="fc-xai", call_id="call-xai", name="hermes_web_search",
            arguments='{"query":"alias dispatch"}',
        )], status="completed")
        normalized = agent._get_transport().normalize_response(response)
        self.assertEqual("web_search", normalized.tool_calls[0].name)
        agent.valid_tool_names = {"web_search", "web_extract"}
        agent._flush_messages_to_session_db = lambda *a, **kw: True
        messages = []
        execute_tool_calls_sequential(agent, normalized, messages, "chain-test", finalize=False)
        self.assertIn('"selected": "xai-client-exa"', messages[-1]["content"])
        search.assert_called_once()

    def test_invalid_shapes_duplicates_and_unknown_names_are_errors(self):
        for value in [None, [], "native,exa", [""], [1], ["native", "native"], ["openai-native"]]:
            with self.subTest(value=value):
                self.config["web"]["search_backends"] = value
                with self.assertRaisesRegex(ValueError, "web.search_backends"):
                    self.agent()
        self.config["web"]["search_backends"] = ["native", "typo-provider"]
        agent = self.agent()
        with self.assertRaisesRegex(ValueError, "typo-provider"):
            agent._build_api_kwargs([{"role": "user", "content": "hello"}])

    def test_missing_credentials_exhausts_chain_without_rescue_or_autodetect(self):
        from tools.registry import registry
        self.config["web"]["search_backends"] = ["native", "exa"]
        self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.is_available", return_value=False))
        rescue = self.enterContext(patch("tools.web_tools._rescue_search"))
        result = json.loads(registry.dispatch("web_search", {"query": "no credentials"}))
        self.assertFalse(result["success"])
        self.assertEqual(2, len(result["diagnostics"]))
        rescue.assert_not_called()

    def test_legacy_single_native_config_keeps_existing_behavior(self):
        self.enterContext(patch("agent.transports.codex._openai_prefers_native_web_search", return_value=True))
        self.enterContext(patch("agent.transports.codex._xai_prefers_native_web_search", return_value=True))
        for key in ("backend", "search_backend"):
            for name in ("openai-native", "xai"):
                with self.subTest(key=key, name=name):
                    self.config["web"] = {key: name}
                    agent = self.agent() if name == "openai-native" else self.agent(
                        "unverified-grok", "xai", "https://api.x.ai/v1")
                    body = agent._build_api_kwargs([{"role": "user", "content": "hello"}])
                    self.assertIn({"type": "web_search"}, body["tools"])
                    self.assertIsNone(agent._web_search_selection)

    def test_actual_switch_model_and_provider_recompute_next_request(self):
        agent = self.agent()
        chain = agent._web_search_chain
        for model, provider, endpoint, native in [
            ("unknown-model", "openai-codex", "https://chatgpt.com/backend-api/codex", False),
            ("grok-composer-2.5-fast", "xai", "https://api.x.ai/v1", True),
            ("gpt-5-codex", "openai-codex", "https://chatgpt.com/backend-api/codex", True),
        ]:
            agent.switch_model(model, provider, api_key="test-placeholder", base_url=endpoint,
                               api_mode="codex_responses")
            body = agent._build_api_kwargs([{"role": "user", "content": "search"}])
            self.assertEqual(native, {"type": "web_search"} in body["tools"])
            self.assertEqual(chain, agent._web_search_chain)

    def test_selection_uses_actual_transport_and_default_profiles_have_no_native(self):
        from agent.web_search_selection import native_search_tool
        from agent.transports import get_transport
        for provider in ("anthropic", "gemini", "openai", "custom"):
            agent = SimpleNamespace(provider=provider, model="gpt-5-codex", api_mode="codex_responses",
                                    base_url="https://chatgpt.com/backend-api/codex",
                                    _get_transport=lambda: get_transport("codex_responses"))
            self.assertIsNone(native_search_tool(agent, agent.model))
        agent = self.agent()
        agent._get_transport = lambda: get_transport("chat_completions")
        self.assertIsNone(native_search_tool(agent, agent.model))

    def test_route_and_tool_overrides_cannot_bypass_selection_or_disabled_tools(self):
        agent = self.agent()
        agent.request_overrides = {"extra_body": {"model": "unknown-model"}}
        with self.assertRaisesRegex(ValueError, "search_backends"):
            agent._build_api_kwargs([{"role": "user", "content": "search"}])
        agent.request_overrides = {"tools": [{"type": "web_search"}]}
        with self.assertRaisesRegex(ValueError, "search_backends"):
            agent._build_api_kwargs([{"role": "user", "content": "search"}], tools_for_api=[])

    def test_unavailable_probe_skips_with_diagnostic(self):
        self.config["web"]["search_backends"] = ["exa", "firecrawl"]
        self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.is_available", side_effect=RuntimeError("probe unavailable")))
        agent = self.agent()
        agent._build_api_kwargs([{"role": "user", "content": "search"}])
        self.assertIsNone(agent._web_search_selection.backend)
        self.assertTrue(any("exa" in d for d in agent._web_search_selection.diagnostics))

    def test_runtime_client_errors_never_rescue_or_advance_chain(self):
        from agent.web_search_selection import search_selection_scope
        from tools.registry import registry
        self.config["web"]["search_backends"] = ["exa", "firecrawl"]
        self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.is_available", return_value=True))
        exa = self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.search", return_value={"success": False, "error": "HTTP 429"}))
        firecrawl = self.enterContext(patch("plugins.web.firecrawl.provider.FirecrawlWebSearchProvider.search"))
        rescue = self.enterContext(patch("tools.web_tools._rescue_search"))
        managed = self.enterContext(patch("tools.web_tools._managed_search_fallback"))
        agent = self.agent()
        agent._build_api_kwargs([{"role": "user", "content": "search"}])
        with search_selection_scope(agent._web_search_selection):
            result = json.loads(registry.dispatch("web_search", {"query": "runtime"}))
        self.assertEqual("HTTP 429", result["error"])
        exa.assert_called_once()
        firecrawl.assert_not_called()
        rescue.assert_not_called()
        managed.assert_not_called()

    def test_vendor_keyless_rescue_cannot_defeat_order_after_credentials_disappear(self):
        from agent.web_search_selection import search_selection_scope
        from tools.registry import registry
        self.config["web"]["search_backends"] = ["exa", "firecrawl"]
        self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.is_available", return_value=True))
        agent = self.agent()
        agent._build_api_kwargs([{"role": "user", "content": "search"}])
        self.enterContext(patch("plugins.web.exa.provider.provider_env", return_value=""))
        ring = self.enterContext(patch("plugins.web.keyless_mcp.search_with_failover"))
        with search_selection_scope(agent._web_search_selection):
            result = json.loads(registry.dispatch("web_search", {"query": "no ring"}))
        self.assertFalse(result["success"])
        ring.assert_not_called()

    def test_native_error_issues_one_model_request_and_no_chain_retry(self):
        from unittest.mock import Mock
        agent = self.agent()
        body = agent._build_api_kwargs([{"role": "user", "content": "search"}])
        create = Mock(side_effect=ValueError("native tool rejected"))
        client = SimpleNamespace(responses=SimpleNamespace(create=create), base_url=agent.base_url)
        rescue = self.enterContext(patch("tools.web_tools._rescue_search"))
        with self.assertRaisesRegex(ValueError, "native tool rejected"):
            agent._run_codex_stream(body, client=client)
        self.assertEqual(1, create.call_count)
        self.assertEqual("native", agent._web_search_selection.backend)
        rescue.assert_not_called()

    def test_extract_uses_its_own_provider_even_under_search_selection(self):
        from agent.web_search_selection import search_selection_scope
        from tools.web_tools import web_extract_tool
        self.config["web"]["extract_backend"] = "exa"
        extract = self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.extract", return_value=[{
            "url": "https://example.com", "title": "Extract", "content": "independent extraction",
        }]))
        self.enterContext(patch("tools.web_tools.async_is_safe_url", return_value=True))
        agent = self.agent()
        agent._build_api_kwargs([{"role": "user", "content": "search"}])
        with search_selection_scope(agent._web_search_selection), closing(asyncio.new_event_loop()) as loop:
            result = json.loads(loop.run_until_complete(web_extract_tool(["https://example.com"])))
        self.assertEqual("independent extraction", result["results"][0]["content"])
        extract.assert_called_once()

    def test_interleaved_contexts_pin_backend_and_survive_thread_propagation(self):
        from concurrent.futures import ThreadPoolExecutor
        from agent.secret_scope import set_secret_scope, reset_secret_scope
        from agent.web_search_selection import search_selection_scope
        from agent.web_search_provider import get_provider_env
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        from tools.thread_context import propagate_context_to_thread
        from tools.registry import registry
        self.config["web"]["search_backends"] = ["native", "exa"]
        self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.search", side_effect=lambda q, n: {
            "success": True, "profile_marker": get_provider_env("EXA_API_KEY"), "data": {"web": []},
        }))

        def build(profile, model, ctx):
            home = self.home / profile
            home.mkdir()
            ht = set_hermes_home_override(home)
            st = set_secret_scope({"EXA_API_KEY": profile}, profile_home=str(home))
            self.addCleanup(ctx.run, reset_secret_scope, st)
            self.addCleanup(ctx.run, reset_hermes_home_override, ht)
            agent = self.agent(model=model)
            agent._build_api_kwargs([{"role": "user", "content": "search"}])
            return agent

        a_context, b_context = contextvars.Context(), contextvars.Context()
        a = a_context.run(build, "profile-a", "unverified", a_context)
        b = b_context.run(build, "profile-b", "gpt-5-codex", b_context)

        def dispatch(agent):
            with search_selection_scope(agent._web_search_selection):
                return propagate_context_to_thread(lambda: json.loads(registry.dispatch("web_search", {"query": "same query"})))

        a_worker = a_context.run(dispatch, a)
        # Rebuild another request before the previously captured worker runs.
        a.model = "gpt-5-codex"
        a_context.run(a._build_api_kwargs, [{"role": "user", "content": "next"}])
        self.assertEqual("native", b._web_search_selection.backend)
        b.model = "unverified"
        b_context.run(b._build_api_kwargs, [{"role": "user", "content": "search"}])
        b_worker = b_context.run(dispatch, b)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(a_worker).result(timeout=10)
            other = pool.submit(b_worker).result(timeout=10)
        self.assertEqual("profile-a", result["profile_marker"])
        self.assertEqual("profile-b", other["profile_marker"])

    def test_concurrent_tool_executor_dispatches_the_declared_client(self):
        from agent.tool_executor import execute_tool_calls_concurrent
        from agent.web_search_selection import current_search_selection
        self.config["web"] = {"search_backends": ["exa"], "search_backend": "firecrawl"}
        self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.is_available", return_value=True))
        search = self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.search", return_value={
            "success": True, "selected": "exa", "data": {"web": []},
        }))
        agent = self.agent()
        agent.valid_tool_names = {"web_search", "web_extract"}
        agent._flush_messages_to_session_db = lambda *a, **kw: True
        agent._build_api_kwargs([{"role": "user", "content": "search"}])
        calls = [SimpleNamespace(id=f"call-{i}", function=SimpleNamespace(
            name="web_search", arguments=json.dumps({"query": f"query-{i}"}))) for i in range(2)]
        messages = []
        execute_tool_calls_concurrent(agent, SimpleNamespace(tool_calls=calls), messages, "chain-test", finalize=False)
        self.assertEqual(2, search.call_count)
        self.assertEqual(2, len(messages))
        self.assertTrue(all('"selected": "exa"' in m["content"] for m in messages))
        self.assertIsNone(current_search_selection())

    def test_real_profile_config_files_are_isolated_and_chain_is_pinned(self):
        from agent.secret_scope import set_secret_scope, reset_secret_scope
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        self.enterContext(patch("hermes_cli.config.load_config_readonly", real_load_config_readonly))
        self.enterContext(patch("hermes_cli.config.load_config", real_load_config))
        for name, chain, expected in [("first", ["native", "exa"], "native"), ("second", ["exa", "native"], "exa")]:
            with self.subTest(profile=name):
                home = self.home / name
                home.mkdir()
                (home / "config.yaml").write_text(json.dumps({"web": {"search_backends": chain}}), encoding="utf-8")
                ht = set_hermes_home_override(home)
                st = set_secret_scope({"EXA_API_KEY": "test-placeholder"}, profile_home=str(home))
                try:
                    agent = self.agent()
                    body = agent._build_api_kwargs([{"role": "user", "content": "search"}])
                    self.assertEqual(expected, agent._web_search_selection.backend)
                    self.assertEqual(expected == "native", {"type": "web_search"} in body["tools"])
                    (home / "config.yaml").write_text(json.dumps({"web": {"search_backends": ["native"]}}), encoding="utf-8")
                    agent._build_api_kwargs([{"role": "user", "content": "next"}])
                    self.assertEqual(tuple(chain), agent._web_search_chain)
                finally:
                    reset_secret_scope(st)
                    reset_hermes_home_override(ht)

    def test_legacy_client_search_keys_and_runtime_rescue_are_preserved(self):
        from tools.registry import registry
        self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.search", return_value={
            "success": False, "error": "HTTP 429",
        }))
        self.enterContext(patch("tools.web_tools._rescue_eligible", return_value=True))
        rescue = self.enterContext(patch("tools.web_tools._rescue_search", return_value={"success": True, "data": {"web": []}, "rescued": True}))
        for config in ({"backend": "exa"}, {"backend": "firecrawl", "search_backend": "exa"}):
            with self.subTest(config=config):
                self.config["web"] = config
                result = json.loads(registry.dispatch("web_search", {"query": "old configuration"}))
                self.assertTrue(result.get("rescued"), result)
        self.assertEqual(2, rescue.call_count)

    def test_legacy_request_does_not_adopt_a_chain_added_after_declaration(self):
        from agent.tool_executor import execute_tool_calls_sequential
        self.config["web"] = {"search_backend": "exa"}
        search = self.enterContext(patch("plugins.web.exa.provider.ExaWebSearchProvider.search", return_value={
            "success": True, "selected": "legacy-exa", "data": {"web": []},
        }))
        agent = self.agent()
        agent.valid_tool_names = {"web_search", "web_extract"}
        agent._flush_messages_to_session_db = lambda *a, **kw: True
        agent._build_api_kwargs([{"role": "user", "content": "search"}])
        self.config["web"]["search_backends"] = ["native"]
        messages = []
        call = SimpleNamespace(id="legacy-call", function=SimpleNamespace(name="web_search", arguments='{"query":"legacy snapshot"}'))
        execute_tool_calls_sequential(agent, SimpleNamespace(tool_calls=[call]), messages, "chain-test", finalize=False)
        self.assertIn('"selected": "legacy-exa"', messages[-1]["content"])
        search.assert_called_once()

    def test_firecrawl_chain_cannot_return_another_profiles_cached_client(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from agent.web_search_selection import search_selection_scope
        from plugins.web.firecrawl import provider as firecrawl
        from tools.thread_context import propagate_context_to_thread
        from tools.web_tools import web_search_tool
        self.config["web"] = {"search_backends": ["firecrawl"]}
        self.enterContext(patch.object(firecrawl, "_env", side_effect=lambda k: "profile-b-key" if k == "FIRECRAWL_API_KEY" else ""))
        ready, finished = threading.Event(), threading.Event()
        primary_thread = threading.current_thread()

        def client(label):
            return SimpleNamespace(search=lambda **kw: {"web": [{
                "title": label, "url": "https://example.com", "description": label,
            }]})

        a_client, b_client = client("profile-a"), client("profile-b")

        class InterleavedCache:
            # The first read snapshots A's client. A concurrent request then
            # installs B's pair before the first reader can inspect the key.
            _firecrawl_client = a_client
            config = ("direct", None, "profile-a-key")

            @property
            def _firecrawl_client_config(self):
                if threading.current_thread() is primary_thread:
                    ready.set()
                    if not finished.wait(10):
                        raise TimeoutError("concurrent client construction did not finish")
                return self.config

            @_firecrawl_client_config.setter
            def _firecrawl_client_config(self, value):
                self.config = value

        def factory(**kwargs):
            ready.set()
            return b_client

        self.enterContext(patch.object(firecrawl, "_wt", return_value=InterleavedCache()))
        self.enterContext(patch.object(firecrawl, "Firecrawl", side_effect=factory))
        agent = self.agent(model="unverified")
        agent._build_api_kwargs([{"role": "user", "content": "search"}])

        def concurrent_request():
            try:
                if not ready.wait(10):
                    raise TimeoutError("primary client request did not start")
                return firecrawl._get_firecrawl_client("search")
            finally:
                finished.set()

        with search_selection_scope(agent._web_search_selection), ThreadPoolExecutor(max_workers=1) as pool:
            other = pool.submit(propagate_context_to_thread(concurrent_request))
            result = json.loads(web_search_tool("profile isolation"))
            self.assertIs(b_client, other.result(timeout=10))
        self.assertEqual("profile-b", result["data"]["web"][0]["title"])


def load_tests(loader, tests, pattern):
    # Keep the offline entry point inclusive of the legacy-cache regressions;
    # pytest collects their own file directly, without duplicating them here.
    from tests.plugins.web.test_firecrawl_client_cache import FirecrawlClientCacheTests
    tests.addTests(loader.loadTestsFromTestCase(FirecrawlClientCacheTests))
    return tests


if __name__ == "__main__":
    unittest.main()
