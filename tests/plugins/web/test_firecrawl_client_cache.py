"""Firecrawl cache must not leak clients between concurrent profile configs."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, local
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from plugins.web.firecrawl import provider
from tools import tool_backend_helpers


class FirecrawlClientCacheTests(unittest.TestCase):
    def setUp(self):
        self.profile = local()
        self.clients = []
        self.enterContext(patch.object(provider, "_get_direct_firecrawl_config", self.config))
        self.enterContext(patch.object(provider, "Firecrawl", self.factory))
        self.enterContext(patch.object(tool_backend_helpers, "read_web_capability_selection", return_value="firecrawl"))

    def config(self):
        key = self.profile.key
        return "sdk", {"api_key": key}, ("direct", None, key)

    def factory(self, **kwargs):
        client = SimpleNamespace(**kwargs)
        self.clients.append(client)
        return client

    def test_concurrent_profiles_return_their_own_constructed_client(self):
        profile = self.profile
        published, resume = Event(), Event()

        class Cache:
            def __setattr__(self, name, value):
                object.__setattr__(self, name, value)
                # Suspend after the first publication, regardless of how many
                # slots the cache uses or whether it publishes a client or pair.
                if value is not None and profile.key == "profile-a" and not published.is_set():
                    published.set()
                    if not resume.wait(10):
                        raise TimeoutError("second profile did not finish")

        self.enterContext(patch.object(provider, "_wt", return_value=Cache()))

        def first_profile():
            profile.key = "profile-a"
            return provider._get_firecrawl_client("search")

        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(first_profile)
            try:
                self.assertTrue(published.wait(10), "first profile did not publish its client")
                profile.key = "profile-b"
                second = provider._get_firecrawl_client("search")
            finally:
                resume.set()
            first_result = first.result(timeout=10)

        self.assertEqual((first_result.api_key, second.api_key), ("profile-a", "profile-b"))
        # Interleaved publication may evict either entry; a subsequent lookup
        # must still use B's config, then reuse it on an unchanged request.
        later = provider._get_firecrawl_client("search")
        self.assertEqual(later.api_key, "profile-b")
        self.assertIs(provider._get_firecrawl_client("extract"), later)

    def test_concurrent_lookup_uses_one_config_client_snapshot(self):
        profile = self.profile
        captured, resume = Event(), Event()
        reading = local()

        class Cache:
            def __getattribute__(self, name):
                value = object.__getattribute__(self, name)
                # Suspend the first populated cache read, without requiring
                # separate config/client slots or a particular slot name.
                if value is not None and getattr(reading, "pause", False):
                    reading.pause = False
                    captured.set()
                    if not resume.wait(10):
                        raise TimeoutError("replacement profile did not finish")
                return value

        self.enterContext(patch.object(provider, "_wt", return_value=Cache()))
        profile.key = "profile-a"
        provider._get_firecrawl_client("extract")

        def lookup():
            profile.key = "profile-b"
            reading.pause = True
            return provider._get_firecrawl_client("extract")

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(lookup)
            try:
                self.assertTrue(captured.wait(10), "lookup did not capture the cache snapshot")
                profile.key = "profile-b"
                replacement = provider._get_firecrawl_client("extract")
            finally:
                resume.set()
            result = pending.result(timeout=10)

        self.assertEqual(result.api_key, "profile-b")
        self.assertEqual(replacement.api_key, "profile-b")

    def test_visible_cache_reset_forces_rebuild(self):
        import tools.web_tools as wt

        self.enterContext(patch.object(wt, "_firecrawl_client", None))
        self.enterContext(patch.object(wt, "_firecrawl_client_config", None))
        self.profile.key = "profile-a"
        first = provider._get_firecrawl_client("search")
        self.assertIs(provider._get_firecrawl_client("extract"), first)

        wt._firecrawl_client = None
        rebuilt = provider._get_firecrawl_client("search")
        self.assertIsNot(rebuilt, first)
        self.assertEqual(rebuilt.api_key, "profile-a")
        self.assertEqual(len(self.clients), 2)


if __name__ == "__main__":
    unittest.main()
