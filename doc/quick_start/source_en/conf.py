# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = u'TPU-MLIR'
copyright = u'2022, SOPHGO'
author = u'SOPHGO'

import os, subprocess, re
command_line = "git describe --tags --always"
tag_str = u""
try:
    tag_str = subprocess.check_output(command_line, shell = True).decode()
except subprocess.TimeoutExpired as time_e:
    print(time_e)
except subprocess.CalledProcessError as call_e:
    print(call_e.output.decode(encoding="utf-8"))
tag_find = re.findall("(\d+)\.(\d+)\-(\d+)", tag_str)
if not tag_find:
    tag_find = re.findall("(\d+)\.(\d+)", tag_str)
    assert(tag_find)
release =  ".".join(tag_find[0])


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = ["../../templates"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../assets']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "TPU-MLIR Quick Start"


# 图片和表格自动编号
numfig = True

# -- Options for LaTeX output ------------------------------------------------
latex_engine = 'xelatex'
latex_elements = {
    # Papersize ('letterpaper' or 'a4paper'), default is letterpaper
    'papersize': 'a4paper',
    # 设置页边距大小
    'geometry': r' \usepackage[left=2.8cm,right=2.6cm,top=3.7cm,bottom=3.5cm]{geometry}',
    # The font size ('10pt', '11pt' or '12pt'), default is '10pt'.
    'pointsize': '11pt',
    # \setCJKmainfont[BoldFont=Heiti SC Medium]{Heiti SC Light}
    # \setCJKmonofont[BoldFont=Times Regular]{Times Italic}
    # `\setmainfont`、`\setsansfont{}`、`\setmonofont{}`
    # 分别设置正文字体、无衬线字体：标题、等宽字体：用于抄录内容
    'fontpkg': r'''
    \setmainfont{FandolSong}
    \setsansfont{FandolHei}
    \setmonofont{FandolFang}
    ''',
    # 设置章节标题样式
    # \usepackage[Lenny]{fncychap}  Bjarne, Sonny, Lenny, Glenn, Conny, Rejne
    'fncychap': '\\usepackage[Sonny]{fncychap}',
    # 图片严格出现在文字处
    # 'figure_align': 'H',
    # preamble 样式
    # 目录样式：tocloft
    # 每节从新页面开始：newcommand{\sectionbreak}{\clearpage}
    # 全文文本左对齐：\usepackage[document]{ragged2e}
    'preamble':r'''
    \usepackage{tocloft}
    \renewcommand\cftfignumwidth{4em}
    \renewcommand\cfttabnumwidth{4em}
    \renewcommand\cftsecnumwidth{4em}
    \renewcommand\cftsubsecnumwidth{6em}
    \renewcommand\cftparanumwidth{6em}
    \usepackage{fancyhdr}
    \setlength\headheight{14pt}
    \fancypagestyle{normal}{
        \fancyhead[R]{}
        \fancyhead[C]{\leftmark}
        \fancyfoot[C]{Copyright © SOPHGO}
        \fancyfoot[R]{\thepage}
        \renewcommand{\headrulewidth}{0.4pt}
        \renewcommand{\footrulewidth}{0pt}
    }
    ''',
    'extraclassoptions': 'openany,oneside',

    # 'classoptions': ',zh_CN',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# latex_logo = '../../common/images/logo.png'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, project + '.tex', htmlhelp_basename,
     author, 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, project, htmlhelp_basename,
     [author], 1)
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
