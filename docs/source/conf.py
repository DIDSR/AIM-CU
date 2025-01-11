# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))

project = 'AIM-CU'
copyright = '2024, Smriti Prathapan, Berkman Sahiner, Dhaval Kadia, Ravi K. Samala'
author = 'Smriti Prathapan, Berkman Sahiner, Dhaval Kadia, Ravi K. Samala'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx.ext.duration',
              'sphinx_tabs.tabs',
              'sphinx_simplepdf',
            #   'sphinxcontrib.inkscapeconverter',
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
# html_static_path = ['_static']

html_show_copyright = False
html_theme_options = {
  'sidebar_width': '300px',
  'page_width': '1200px',
  'body_max_width': 'auto',
  'github_button': 'true',
  'github_user': 'DIDSR',
  'github_repo': 'AIM-CU',
}
# html_sidebars = {'**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html']}
