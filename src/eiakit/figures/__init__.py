"""
Figures: one description (spec.Figure), two renderers.
    render_mpl.to_matplotlib(fig)  -> matplotlib Figure (PRL style; PDF / PNG)
    render_plotly.to_plotly(fig)   -> plotly Figure (interactive HTML: zoom, legend toggle)
    export.save(fig, base)         -> base.pdf, base.png, base.html
"""
from .spec import Band, ErrorBars, Figure, Line, Panel, Points, Text, lab  # noqa: F401
