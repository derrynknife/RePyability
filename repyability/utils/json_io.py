"""Reading and writing JSON documents as surpyval does: ``to_json(fp)``
writes to a path, and ``from_json`` reads a path, as well as the text."""

import os
from pathlib import Path
from typing import Any, Optional


def write_json(text: str, fp: Any = None) -> Optional[str]:
    """``text`` (a JSON document) itself, with no ``fp``; or written to
    ``fp``, a path or a file opened for writing, and None."""
    if fp is None:
        return text
    if hasattr(fp, "write"):
        fp.write(text)
        return None
    with open(fp, "w", encoding="utf-8") as file:
        file.write(text)
    return None


def read_json(source: Any) -> str:
    """The JSON document ``source`` gives: the text itself, a path to a
    file holding it (``str`` or ``os.PathLike``), or a file opened for
    reading. A string is the text if it starts (after any white space) with
    ``{`` or ``[``, and a path otherwise."""
    if hasattr(source, "read"):
        return source.read()
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    if isinstance(source, str) and source.lstrip().startswith(("{", "[")):
        return source
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(
                f"{str(source)!r} is neither a JSON document nor a file."
            )
        return path.read_text(encoding="utf-8")
    raise TypeError(
        "Give a JSON document as its text, a path to a file, or a file, "
        f"not a {type(source).__name__}."
    )
