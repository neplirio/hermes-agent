"""Conservative hosted-search capability rules owned by LLM provider profiles."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeSearchRule:
    endpoint: str
    models: tuple[str, ...]
    api_mode: str = "codex_responses"
    transport: str = "agent.transports.codex.ResponsesApiTransport"
    tool_type: str = "web_search"

    def matches(self, *, endpoint: str, model: str, api_mode: str, transport: str) -> bool:
        # Exact origin AND path: a model name, provider label or compatible proxy
        # alone cannot establish that this endpoint hosts the tool.
        return (
            endpoint.rstrip("/") == self.endpoint
            and model in self.models
            and api_mode == self.api_mode
            and transport == self.transport
        )
