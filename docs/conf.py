# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = u"stentfit"
copyright = u"2026, Vural Aktas"
author = u"Vural Aktas"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]
autoapi_dirs = ["../src"]

# Render Mermaid diagrams at their natural aspect ratio (not squashed into a
# fixed-height box) and let readers pan/zoom the busier workflow diagrams.
mermaid_height = "auto"
mermaid_d3_zoom = True
mermaid_init_config = {
    "startOnLoad": True,
    "theme": "neutral",
    "themeVariables": {"fontSize": "18px"},
}

# The example notebooks are pre-executed copies (symlinked from examples/):
# the real pipeline needs an STL file that isn't in the repo (data/ is
# git-ignored) and calls GMSH / long-running skeletonisation, so it isn't
# safe or fast to re-execute on every docs build. Their stored outputs are
# rendered as-is.
nb_execution_mode = "off"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
