# Copyright (c) ICU-Guardian Contributors
# Licensed under MIT License

"""
ICU-Guardian Environment Client.

Subclasses OpenEnv's HTTPEnvClient for type-safe interaction with the
ICU-Guardian environment from the agent side.
"""

from models import ICUAction, ICUObservation

# Support both in-repo and standalone imports
try:
    from openenv.core.http_env_client import HTTPEnvClient
except ImportError:
    from openenv_core.http_env_client import HTTPEnvClient


class ICUGuardianEnv(HTTPEnvClient):
    """
    Client for interacting with the ICU-Guardian environment.

    Usage (async):
        async with ICUGuardianEnv(base_url="https://singhhrishabh-icu-guardian.hf.space") as env:
            result = await env.reset()
            result = await env.step(ICUAction(action="adjust_oxygen", level="increase"))
            print(result.HR, result.reward)

    Usage (sync):
        with ICUGuardianEnv(base_url="...").sync() as env:
            result = env.reset()
            result = env.step(ICUAction(action="wait"))
    """

    ACTION_CLASS = ICUAction
    OBSERVATION_CLASS = ICUObservation
