"""
Moving Impact model server.

A thin local wrapper around the scripts that already exist: it serves the HTML
tools over the same URL space Live Server does, and runs the Python scripts as
subprocesses through the --scenario command line that they already accept.

Deliberately not a reimplementation. Everything the server runs, you can run
yourself from a terminal with the same arguments, and the reverse is true too --
there is one code path, not two.
"""

__all__ = ["jobs", "tasks", "api"]
