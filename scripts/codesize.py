#!/usr/bin/env python3
"""Per-module machine-code size of fastlanes, as a markdown table.

Builds the release test binary with debug info: the tests call every kernel with concrete types,
so the binary holds each kernel's machine code. The debug info records every function, including
each copy inlined into another function, under its module path, so a library function counts
towards its `fastlanes::<module>` wherever its code ends up, even inlined into a test. Test code
and other crates are not counted.

Usage (needs `llvm-dwarfdump` from LLVM, e.g. `apt install llvm` or `brew install llvm`):
    scripts/codesize.py                # the working tree
    scripts/codesize.py --base develop # the working tree, compared with a git revision
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

HEADER = re.compile(r"^0x([0-9a-f]+):( +)(DW_TAG_\w+|NULL)")
ATTR = re.compile(r"^\s+DW_AT_(name|abstract_origin|specification|low_pc|high_pc|ranges)\s+\((.*)")
RANGE = re.compile(r"\[0x([0-9a-f]+), 0x([0-9a-f]+)\)")


def build(repo: str) -> str:
    """Build the release test binary of the crate in `repo` and return its path."""
    # One target directory for every revision, so the dependencies are only built once.
    env = dict(os.environ, CARGO_PROFILE_RELEASE_DEBUG="limited",
               CARGO_TARGET_DIR=os.environ.get("CARGO_TARGET_DIR", os.path.abspath("target")))
    cmd = ["cargo", "test", "--release", "--features", "runtime", "--lib", "--no-run",
           "--message-format=json"]
    out = subprocess.run(cmd, cwd=repo, env=env, check=True, stdout=subprocess.PIPE, text=True)
    for line in out.stdout.splitlines():
        msg = json.loads(line)
        if msg.get("reason") == "compiler-artifact" and msg["profile"]["test"] and msg["executable"]:
            return msg["executable"]
    sys.exit("no test binary was built")


def dwarfdump() -> str:
    if path := shutil.which("llvm-dwarfdump"):
        return path
    # Versioned (Debian/Ubuntu) and keg-only (Homebrew) LLVM installs.
    for pattern in ("/usr/lib/llvm-*/bin/llvm-dwarfdump", "/usr/local/opt/llvm/bin/llvm-dwarfdump",
                    "/opt/homebrew/opt/llvm/bin/llvm-dwarfdump"):
        if found := sorted(glob.glob(pattern)):
            return found[-1]
    sys.exit("llvm-dwarfdump not found: install LLVM (`apt install llvm` or `brew install llvm`)")


def parse(binary: str) -> dict:
    """The debug info entries of `binary`, keyed by offset."""
    out = subprocess.run([dwarfdump(), "--debug-info", binary], check=True, capture_output=True,
                         text=True).stdout
    dies, stack, cur = {}, [], None  # stack holds (indent, offset) of the open entries
    for line in out.splitlines():
        if m := HEADER.match(line):
            while stack and stack[-1][0] >= len(m[2]):
                stack.pop()
            if m[3] == "NULL":
                cur = None
                continue
            parent = stack[-1][1] if stack else None
            cur = dies[int(m[1], 16)] = {"tag": m[3], "parent": parent, "children": [],
                                         "ranges": [], "name": None, "origin": None}
            if parent is not None:
                dies[parent]["children"].append(int(m[1], 16))
            stack.append((len(m[2]), int(m[1], 16)))
            continue
        if cur is None:
            continue
        if m := ATTR.match(line):
            key, val = m[1], m[2]
            if key == "name":
                cur["name"] = val.split('"')[1]
            elif key in ("abstract_origin", "specification"):
                cur["origin"] = int(val.split()[0], 16)
            elif key == "low_pc":
                cur["low"] = int(val.rstrip(")"), 16)
            elif key == "high_pc":
                cur["ranges"].append((cur.pop("low"), int(val.rstrip(")"), 16)))
            elif key == "ranges":
                cur["in_ranges"] = True
        if cur.get("in_ranges"):  # a range list spans several lines
            cur["ranges"] += [(int(a, 16), int(b, 16)) for a, b in RANGE.findall(line)]
            cur["in_ranges"] = not line.rstrip().endswith("))")
    return dies


def module(dies: dict, off: int) -> str | None:
    """`delta` for a function in `fastlanes::delta`; None for test code and other crates."""
    while dies[off]["origin"] is not None:
        off = dies[off]["origin"]
    path, p = [], dies[off]["parent"]
    while p is not None:
        if dies[p]["tag"] == "DW_TAG_namespace":
            path.insert(0, dies[p]["name"])
        p = dies[p]["parent"]
    if not path or path[0] != "fastlanes" or {"test", "tests"} & set(path):
        return None
    if len(path) == 1:
        return None if dies[off]["name"] == "main" else "lib"
    return path[1]


def sizes(binary: str) -> Counter:
    dies = parse(binary)
    modules = Counter()

    def count(off: int) -> None:
        """Count `off` if it is library code, otherwise the library code inlined into it."""
        if mod := module(dies, off):
            modules[mod] += sum(b - a for a, b in dies[off]["ranges"])
            return
        for child in dies[off]["children"]:
            count(child)

    for off, die in dies.items():
        if die["tag"] == "DW_TAG_subprogram" and die["ranges"]:
            count(off)
    return modules


def measure_rev(rev: str) -> Counter | None:
    """Sizes at git revision `rev`, or None if it does not build."""
    with tempfile.TemporaryDirectory() as tmp:
        worktree = os.path.join(tmp, "base")
        subprocess.run(["git", "worktree", "add", "-q", "--detach", worktree, rev], check=True)
        try:
            return sizes(build(worktree))
        except subprocess.CalledProcessError:
            return None
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", worktree], check=True)


def kib(n: int) -> str:
    return f"{n / 1024:.1f} KiB"


def change(head: int, base: int) -> str:
    if head == base:
        return "-"
    pct = f" ({(head - base) / base:+.1%})" if base else ""
    return f"{head - base:+,} B{pct}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="git revision to compare with")
    args = parser.parse_args()

    head = sizes(build("."))
    base = measure_rev(args.base) if args.base else None
    print("## Code size\n")
    if base is None:
        print("| Module | Size |\n|---|---:|")
    else:
        print("| Module | Size | Base | Change |\n|---|---:|---:|---:|")
    for mod in sorted(head.keys() | (base or {}).keys(), key=lambda m: -head[m]):
        row = f"| `{mod}` | {kib(head[mod])} |"
        if base is not None:
            row += f" {kib(base[mod])} | {change(head[mod], base[mod])} |"
        print(row)
    total = f"| **Total** | **{kib(head.total())}** |"
    if base is not None:
        total += f" {kib(base.total())} | {change(head.total(), base.total())} |"
    print(total)
    if args.base and base is None:
        print(f"\nThe base revision `{args.base}` did not build, so there is no comparison.")
    print("\n<sub>Machine code per module, including library code inlined into the tests (release "
          "test binary, `runtime` feature, x86_64). Run `scripts/codesize.py --base <rev>` to "
          "reproduce.</sub>")


if __name__ == "__main__":
    main()
