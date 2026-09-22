"""Reference solution for the stencil suite task."""
from .errors import TemplateSyntaxError, UndefinedError
from .render import Template, render

__all__ = ["Template", "TemplateSyntaxError", "UndefinedError", "render"]
