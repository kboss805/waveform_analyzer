# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add project root to sys.path so Sphinx can import modules
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'Waveform Analyzer'
copyright = '2025, Kevin Bossoletti'
author = 'Kevin Bossoletti'
release = '1.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = []

# -- autodoc configuration --------------------------------------------------

autodoc_member_order = 'bysource'
autodoc_mock_imports = [
    'customtkinter',
    'CTkMenuBar',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends._backend_tk',
    'matplotlib.figure',
    'tkinter',
    'numpy',
    'scipy',
]

# -- Napoleon configuration (Google-style docstrings) ------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
