"""Sphinx configuration for pytest-quantum docs."""
from __future__ import annotations

project = "pytest-quantum"
copyright = "2026, Tejas"
author = "Tejas"
release = "0.3.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "pytest-quantum"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_returns = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]

# Suppress minor RST formatting warnings from napoleon's docstring conversion
suppress_warnings = ["docutils.nodes.definition_list"]
