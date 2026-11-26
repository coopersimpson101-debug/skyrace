#!/usr/bin/env python3
"""Build the SITL shared library from a skyforge-fc checkout.

    python scripts/build_sitl.py --skyforge-fc ../skyforge-fc

Compiler: $CC if set, else ``python -m ziglang cc`` if the ziglang package is
importable, else ``cc``. Output: build/sitl/skyrace_sitl.{dll,so,dylib}.
"""
import argparse
import glob
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = ".dll" if sys.platform.startswith("win") else (".dylib" if sys.platform == "darwin" else ".so")


def find_skyforge(arg):
    for c in (arg, os.environ.get("SKYFORGE_FC_DIR"), os.path.join(ROOT, "..", "skyforge-fc"),
              os.path.join(ROOT, "skyforge-fc")):
        if c and os.path.exists(os.path.join(c, "sim", "quad_sim.c")):
            return os.path.abspath(c)
    sys.exit("skyforge-fc checkout not found (pass --skyforge-fc or set SKYFORGE_FC_DIR)")


def compiler():
    if os.environ.get("CC"):
        return shlex.split(os.environ["CC"])
    try:
        import ziglang  # noqa: F401
        cache = os.path.join(ROOT, "build", "zig-cache")
        os.environ.setdefault("ZIG_GLOBAL_CACHE_DIR", cache)
        os.environ.setdefault("ZIG_LOCAL_CACHE_DIR", cache)
        return [sys.executable, "-m", "ziglang", "cc"]
    except ImportError:
        return ["cc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skyforge-fc", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "sitl", "skyrace_sitl" + EXT))
    a = ap.parse_args()
    sf = find_skyforge(a.skyforge_fc)
    core = sorted(glob.glob(os.path.join(sf, "src", "core", "*.c")))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cmd = compiler() + ["-std=c11", "-O2", "-Wall", "-Wextra", "-DFC_HOST", "-shared", "-fPIC",
                        "-I", os.path.join(sf, "sim"), "-I", os.path.join(sf, "src", "core"),
                        "-o", a.out, os.path.join(ROOT, "csrc", "skyrace_sitl.c")] + core + \
        ([] if sys.platform.startswith("win") else ["-lm"])
    print("building", os.path.relpath(a.out, ROOT), "from", len(core), "skyforge-fc core sources")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print("ok")


if __name__ == "__main__":
    main()
