# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# (Useful if we want to host C++ docs on readthedocs.io)
# from m2r2 import MdInclude
# from recommonmark.transform import AutoStructify

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'NeoML'
copyright = '2024, ABBYY'
author = 'ABBYY'


# -- Pre-process root README.md ----------------------------------------------
# (Useful if we want to host C++ docs on readthedocs.io)

# with open('../../../README.md', 'r', encoding='utf-8', newline='\n') as file_in:
#     with open('README.md', 'w', encoding='utf-8', newline='\n') as file_out:
#         for line in file_in:
#             line = line.replace('NeoML_logo.png', 'NeoML_logo_help.png')
#             line = line.replace('NeoML/docs/en/', '')
#             line = line.replace('NeoML/docs/images/', '../images/')
#             file_out.write(line)


# -- Remove links to the heading in another markdown files -------------------
# (Useful if we want to host C++ docs on readthedocs.io)

# from os import walk
# from os.path import join, splitext
# import sys

# Fix links to the external md headings for html help
# def replace_md_with_html(filepath):
#     with open(filepath, 'r', encoding='utf8', newline='\n') as file_in:
#         lines = file_in.readlines()
#     modified = False
#     for i, line in enumerate(lines):
#         if line.find('.md#') != -1:
#             modified = True
#             lines[i] = line.replace('.md#', '.html#')
#     if modified:
#         with open(filepath, 'w', encoding='utf8', newline='\n') as file_out:
#             file_out.writelines(lines)

# Remove links to the external md headings
# def remove_heading_links(filepath):
#     with open(filepath, 'r', encoding='utf8', newline='\n') as file_in:
#         lines = file_in.readlines()
#     modified = False
#     for i in range(len(lines)):
#         pos = lines[i].find('.md#')
#         while pos != -1:
#             modified = True
#             bracket = lines[i].find(')', pos)
#             lines[i] = lines[i][:pos+3] + lines[i][bracket:]
#             pos = lines[i].find('.md#')
#     if modified:
#         with open(filepath, 'w', encoding='utf8', newline='\n') as file_out:
#             file_out.writelines(lines)

# is_html_help = any(map(lambda x: x == 'html', sys.argv))

# for dirpath, _, filenames in walk('.'):
#     for filename in filenames:
#         if splitext(filename)[1] != '.md':
#             continue # skip not markdown files
#         filepath = join(dirpath, filename)
#         if is_html_help:
#             replace_md_with_html(filepath)
#         else:
#             remove_heading_links(filepath)

# -- General configuration ---------------------------------------------------

#import sphinx_rtd_theme
from os import getenv

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_copybutton',   # Add 'copy to clipboard' button to every code snippet
    'sphinx.ext.autodoc',  # Support docstrings from python code
    'sphinx.ext.mathjax',  # Support LaTeX formulas in docs (and docstrings)
#    'sphinx.ext.autosectionlabel',
    'sphinx_rtd_theme',  # Pretty theme for HTML
#     'recommonmark',
    'nbsphinx',  # Support .ipynb files
]

# autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

if getenv('READTHEDOCS') == 'True':
    # readthedocs.io forbids byte-compilation of C-binaries
    # we've got to mock them :(
    autodoc_mock_imports = ['neoml.PythonWrapper']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# source_suffix = ['.rst', '.md']

# Configuring mathjax

mathjax_config = {
    'extensions': ['tex2jax.js'],
    'jax': ['input/TeX', 'output/HTML-CSS'],
}

# Configuring nbsphinx

# readthedocs doesn't allow us to build C++ wrappers
# that's why execution is impossible anyway
nbsphinx_execute = 'never'

# https://github.com/rtfd/recommonmark/blob/master/docs/conf.py
def setup(app):
    config = {
        #'url_resolver': lambda url: print(url),
        'auto_toc_tree_section': 'Contents',
        'enable_eval_rst': False,
        'enable_auto_toc_tree': True,
        #'known_url_schemes': ['http', 'https', 'mailto'],
    }
    # app.add_config_value('recommonmark_config', config, True)
    # app.add_transform(AutoStructify)

    # from m2r to make `mdinclude` work
    # app.add_config_value('no_underscore_emphasis', False, 'env')
    # app.add_config_value('m2r_parse_relative_links', True, 'env')
    # app.add_config_value('m2r_anonymous_references', False, 'env')
    # app.add_config_value('m2r_disable_inline_math', False, 'env')
    # app.add_directive('mdinclude', MdInclude)