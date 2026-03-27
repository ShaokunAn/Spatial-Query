import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'SpatialQuery'
copyright = '2024, Shaokun An'
author = 'Shaokun An'
release = '0.0.2'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'myst_parser',
]

# Mock C++ extension so autodoc can import the package without compiling it
autodoc_mock_imports = ['SpatialQueryEliasFanoDB']

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'special-members': '__init__',
    'show-inheritance': True,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable/', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/stable/', None),
}

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
