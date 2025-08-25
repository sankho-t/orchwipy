import sys


def py_lines(lines: list[str]):
    in_py = False
    for ln in lines:
        if ln.startswith("```python"):
            in_py = True
        elif in_py:
            if ln.startswith("```"):
                return
            yield ln


with open(sys.argv[1], "r") as f:
    for pyln in py_lines(f.readlines()):
        sys.stdout.write(pyln)
