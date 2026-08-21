"""Source root for the benchmark data generation pipeline.

Kept deliberately empty of re-exports: importing :mod:`src.generator` pulls in
the LLM bridge, and ``src`` is also the import root for the CLI shim, so a
side-effecting package init here would run on every invocation.
"""
