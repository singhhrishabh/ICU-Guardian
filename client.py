"""
ICU-Guardian: Environment Client

Subclasses OpenEnv's EnvClient for type-safe interaction with the
ICU-Guardian environment from the agent side.
"""

from models import ICUAction, ICUObservation

try:
    from openenv_core.http_env_client import HTTPEnvClient

    class ICUGuardianEnv(HTTPEnvClient):
        """
        Client for interacting with the ICU-Guardian environment.

        Usage (async):
            async with ICUGuardianEnv(base_url="https://your-space.hf.space") as env:
                result = await env.reset()
                result = await env.step(ICUAction(action="wait"))

        Usage (sync):
            with ICUGuardianEnv(base_url="...").sync() as env:
                result = env.reset()
        """

        ACTION_CLASS = ICUAction
        OBSERVATION_CLASS = ICUObservation

except (ImportError, TypeError):
    # Fallback client for local development
    import httpx
    from dataclasses import asdict

    class ICUGuardianEnv:
        """Fallback HTTP client for local development."""

        def __init__(self, base_url: str = "http://localhost:7860"):
            self.base_url = base_url.rstrip("/")
            self._client = None

        async def __aenter__(self):
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30)
            return self

        async def __aexit__(self, *args):
            if self._client:
                await self._client.aclose()

        async def reset(self):
            resp = await self._client.post("/reset", json={})
            return resp.json()

        async def step(self, action: ICUAction):
            resp = await self._client.post("/step", json=asdict(action))
            return resp.json()

        async def state(self):
            resp = await self._client.get("/state")
            return resp.json()

        async def close(self):
            if self._client:
                await self._client.aclose()
