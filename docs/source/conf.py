"""Documentation Configuration File."""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, os.path.abspath("../../nbed"))


# -- Project information -----------------------------------------------------

project = "Nbed"
copyright = "2022, Michael Williams de la Bastida, Alexis Ralli"
author = "Michael Williams de la Bastida, Alexis Ralli"

# The full version, including alpha/beta/rc tags
release = "0.0.9"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
    # "sphinx.ext.doctest",
    # "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    # "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    # "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "myst_nb",
    "pydata_sphinx_theme",
    "myst_sphinx_gallery",
    ]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

nb_scroll_outputs = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
