"""
Why does PySide6 not load?  Read-only diagnosis (changes nothing).

    python tools\\diagnose_qt.py              diagnosis only
    python tools\\diagnose_qt.py --try-pin    also test, in a throw-away venv under %TEMP%,
                                             whether PySide6 < 6.8 loads with this Python
                                             (downloads ~70 MB; the project .venv is untouched)

Background: PySide6 / Qt >= 6.8 are built with Visual Studio 2022 17.10+, which needs
msvcp140.dll >= 14.40. A venv runs its base python.exe, so Windows takes msvcp140.dll from
the base Python's folder first (Anaconda ships an older one) -> "DLL load failed ... The
specified procedure could not be found".
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def file_version(path: Path) -> str | None:
    if os.name != "nt" or not path.is_file():
        return None
    ver = ctypes.windll.version
    size = ver.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    ver.GetFileVersionInfoW(str(path), 0, size, buf)
    p, n = ctypes.c_void_p(), ctypes.c_uint()
    ver.VerQueryValueW(buf, "\\", ctypes.byref(p), ctypes.byref(n))
    ffi = ctypes.cast(p, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
    ms, ls = ffi[2], ffi[3]
    return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"


def major_minor(v: str | None):
    return tuple(int(x) for x in v.split(".")[:2]) if v else None


def try_import(python: str) -> tuple[bool, str]:
    r = subprocess.run([python, "-c", "import PySide6, PySide6.QtCore as q; "
                        "print(PySide6.__version__, q.qVersion())"],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stdout.strip() or r.stderr.strip().splitlines()[-1]
                               if (r.stdout or r.stderr) else "")


def main():
    print(f"python executable : {sys.executable}")
    print(f"base Python       : {sys.base_prefix}")
    conda = any(s in sys.base_prefix.lower() for s in ("conda", "anaconda", "miniconda"))
    print(f"conda-based       : {conda}")
    try:
        import importlib.metadata as md
        print(f"PySide6 installed : {md.version('PySide6')}")
    except Exception:
        print("PySide6 installed : no")
    ok, msg = try_import(sys.executable)
    print(f"PySide6 import    : {'OK  ' + msg if ok else 'FAILED  ' + msg}")

    base_dll = Path(sys.base_prefix) / "msvcp140.dll"
    sys_dll = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "msvcp140.dll"
    vb, vs = file_version(base_dll), file_version(sys_dll)
    print(f"msvcp140.dll base : {vb or 'not present'}  ({base_dll})")
    print(f"msvcp140.dll sys32: {vs or 'not present'}")
    def full(v):
        return tuple(int(x) for x in v.split(".")) if v else None
    cause = not ok and vb is not None and (major_minor(vb) < (14, 40)
                                           or (vs is not None and full(vb) < full(vs)))
    if cause:
        print("\n=> LIKELY CAUSE: the base Python folder has msvcp140.dll "
              f"{vb}, older than the system one ({vs}) and loaded first; recent Qt "
              "builds need the newer runtime.")
    elif ok:
        print("\n=> PySide6 loads: nothing to fix.")
    else:
        print("\n=> msvcp140.dll is not the (only) cause; send this output.")

    print("\nOther Pythons on this PC (py launcher):")
    py = shutil.which("py")
    others = []
    if py:
        r = subprocess.run([py, "-0p"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split()
            if parts and parts[-1].lower().endswith("python.exe"):
                others.append(parts[-1])
                print("   ", line.strip())
    else:
        print("    (no py launcher)")
    good = [p for p in others if "conda" not in p.lower()
            and Path(p).parent != Path(sys.base_prefix)]
    for p in good:
        r = subprocess.run([p, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                           capture_output=True, text=True)
        print(f"    non-conda Python {r.stdout.strip()} at {p}")

    if "--try-pin" in sys.argv:
        d = Path(tempfile.mkdtemp(prefix="eia_qt_test_"))
        print(f"\nTesting PySide6 < 6.8 in a throw-away venv: {d}")
        subprocess.run([sys.executable, "-m", "venv", str(d)], check=True)
        vpy = str(d / ("Scripts" if os.name == "nt" else "bin") / "python")
        subprocess.run([vpy, "-m", "pip", "install", "-q", "PySide6-Essentials<6.8"],
                       check=False)
        ok2, msg2 = try_import(vpy)
        print(f"PySide6 < 6.8     : {'OK  ' + msg2 if ok2 else 'FAILED  ' + msg2}")
        shutil.rmtree(d, ignore_errors=True)
        if ok2:
            print("=> pinning works: no new Python needed (see the advice below).")

    print("\nOptions, least invasive first:")
    if good:
        print(f"  A. use the existing Python {good[0]}:\n"
              f"       conda deactivate;  . .\\env.ps1 -Recreate -Python \"{good[0]}\"")
    print("  B. keep this Python, pin Qt below 6.8 (reversible):\n"
          "       pip install \"PySide6<6.8\"      (undo: pip install -U PySide6)")
    print("  C. install Python 3.12 from python.org, then . .\\env.ps1 -Recreate")


if __name__ == "__main__":
    main()
