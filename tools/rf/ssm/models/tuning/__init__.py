"""
models/tuning/ — Auto-tuning UI + physics-informed sweep-range helpers.

Split out of the former (monolithic) models/base_ui.py.  This subpackage
has no public API of its own; everything is re-exported through
models/base_ui/__init__.py so `from .base_ui import X` keeps working
unchanged for every call site in the repo.
"""
