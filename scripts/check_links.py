#!/usr/bin/env python3
"""Fail if any relative link or image path in the Markdown files does not resolve."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"""(?:\]\(|src=")([^)"#\s]+)""")


def main():
    bad = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in (".git", "build", "node_modules", "__pycache__")]
        for f in fn:
            if not f.endswith(".md"):
                continue
            p = os.path.join(dp, f)
            for m in LINK.finditer(open(p, encoding="utf-8").read()):
                target = m.group(1)
                if re.match(r"^[a-z]+:", target):
                    continue
                if not os.path.exists(os.path.normpath(os.path.join(dp, target))):
                    bad.append(f"{os.path.relpath(p, ROOT)} -> {target}")
    for b in bad:
        print("broken:", b)
    print(f"{'FAIL' if bad else 'ok'}: {len(bad)} broken relative links")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
