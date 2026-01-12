"""Local OpenAI-compatible load balancer + mock backends.

Designed to be dependency-free (standard library only) so it can run on any dev
machine, while still being able to proxy to real sglang-jax OpenAI servers.
"""

