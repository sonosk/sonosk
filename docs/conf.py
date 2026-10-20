import sonosk

project = "SonoSK"
author = "Soheil Kooklan"
copyright = "2026, Soheil Kooklan"
release = sonosk.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_sitemap",
    "sphinxext.opengraph",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3

napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_member_order = "bysource"

html_baseurl = "https://sonosk.github.io/sonosk/"
sitemap_url_scheme = "{link}"
html_theme = "furo"
html_title = "SonoSK"
html_theme_options = {
    "source_repository": "https://github.com/sonosk/sonosk/",
    "source_branch": "main",
    "source_directory": "docs/",
}

ogp_site_url = "https://sonosk.github.io/sonosk/"
ogp_site_name = "SonoSK"
ogp_image = "https://sonosk.github.io/sonosk/_images/bmode_phantom.png"
ogp_description_length = 200
