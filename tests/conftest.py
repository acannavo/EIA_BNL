"""Tests compare with the legacy files: never read a CBS fit store of the user's machine."""
import os
import tempfile

os.environ["EIA_FITS_ROOT"] = os.path.join(tempfile.gettempdir(), "eiakit-tests-no-fits")
