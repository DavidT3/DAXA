#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 22/03/2023, 11:15. Copyright (c) The Contributors

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'Democratising Archival X-ray Astronomy (DAXA)'
copyright = '2023, The Contributors'
author = 'David J Turner, Jessica E Pilling, Agrim Gupta'

# The full version, including alpha/beta/rc tags
# release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.ifconfig',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.graphviz',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.napoleon',
              'nbsphinx',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.autosummary',
              'sphinx.ext.mathjax',
              'sphinx_rtd_theme']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# This should turn off including typehints in the function signatures in autodoc. That information is already in 
#  the docstring and can look extremely confusing
autodoc_typehints = 'none'
# This will make sure the classes aren't sorted in alphabetical order
autodoc_member_order = 'bysource'

# This should make nbsphinx highlight notebooks better
highlight_language = 'ipython3'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = "_static/daxa-high-resolution-logo-black-on-white-background.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}


# autodoc_mock_imports = ["fitsio", 'regions', 'corner', 'emcee', 'abel']

