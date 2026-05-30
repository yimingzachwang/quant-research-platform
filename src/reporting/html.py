"""Minimal markdown-to-HTML converter for experiment reports.

Intentionally fixed-scope: handles only the exact markdown constructs
produced by src/reporting/markdown.py.  It is NOT a general markdown
engine and must not grow into one.

Supported constructs (all that the reporting layer generates):
    # Heading 1        →  <h1>
    ## Heading 2       →  <h2>
    ### Heading 3      →  <h3>
    ---                →  <hr>
    | col | col |      →  <table> with <thead>/<tbody>
    |---|---|           →  table separator row (consumed, not rendered)
    ![alt](path)       →  <figure><img>
    **bold**           →  <strong>  (inline)
    `code`             →  <code>    (inline)
    * item / - item    →  <ul><li>
    blank line         →  close current block
    other text         →  <p>

No nested markdown, no recursive parsing, no template system, no plugins.
stdlib only — no external dependencies.
"""

from __future__ import annotations

import html as _html_lib
import re

_CSS = """\
body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#222;line-height:1.6}
h1,h2,h3{margin-top:1.5em;border-bottom:1px solid #e8e8e8;padding-bottom:.25em}
table{border-collapse:collapse;width:100%;margin:1em 0}
th,td{border:1px solid #ccc;padding:8px 12px;text-align:left}
th{background:#f5f5f5;font-weight:600}
code{background:#f0f0f0;padding:2px 5px;border-radius:3px;font-size:.88em;font-family:monospace}
img{max-width:100%;height:auto;display:block}
figure{margin:1.5em 0}
figcaption{color:#666;font-size:.85em;margin-top:6px}
hr{border:none;border-top:1px solid #ddd;margin:2em 0}
ul{padding-left:1.5em}
li{margin:.3em 0}
p{margin:.75em 0}"""


def markdown_to_html(md: str, title: str = "") -> str:
    """Convert a reporting-layer markdown string to a minimal standalone HTML page.

    Only handles the constructs listed in the module docstring.  Input that
    does not match a known construct is treated as paragraph text.

    Args:
        md: Markdown string produced by src/reporting/markdown.py.
        title: Text placed in the HTML <title> tag.

    Returns:
        Complete standalone HTML string (UTF-8, no external dependencies).
    """
    body = _convert_body(md.splitlines())
    return _wrap_html("\n".join(body), title)


# ---------------------------------------------------------------------------
# Body converter — line-by-line state machine
# ---------------------------------------------------------------------------


def _convert_body(lines: list[str]) -> list[str]:
    """Convert markdown lines to HTML lines via a simple state machine.

    States: "none" | "p" | "ul" | "table_head" | "table_body"
    """
    out: list[str] = []
    state = "none"

    for raw in lines:
        line = raw.rstrip()

        # --- Table rows ---
        if line.startswith("|"):
            if _is_table_separator(line):
                # Transition thead → tbody
                if state == "table_head":
                    out.append("</tr></thead><tbody>")
                    state = "table_body"
                continue  # separator row is consumed

            if state not in ("table_head", "table_body"):
                out.extend(_close_block(state))
                out.append("<table><thead>")
                state = "table_head"

            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if state == "table_head" else "td"
            row = "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells)
            out.append(f"<tr>{row}</tr>")
            continue

        # Leaving a table — close it before processing the current line
        if state in ("table_head", "table_body"):
            out.extend(_close_block(state))
            state = "none"

        # --- Non-table constructs ---
        if line == "---":
            out.extend(_close_block(state))
            state = "none"
            out.append("<hr>")

        elif line.startswith("# "):
            out.extend(_close_block(state))
            state = "none"
            out.append(f"<h1>{_inline(line[2:])}</h1>")

        elif line.startswith("## "):
            out.extend(_close_block(state))
            state = "none"
            out.append(f"<h2>{_inline(line[3:])}</h2>")

        elif line.startswith("### "):
            out.extend(_close_block(state))
            state = "none"
            out.append(f"<h3>{_inline(line[4:])}</h3>")

        elif line.startswith("!["):
            out.extend(_close_block(state))
            state = "none"
            out.append(_parse_image(line))

        elif line.startswith("* ") or line.startswith("- "):
            if state != "ul":
                out.extend(_close_block(state))
                out.append("<ul>")
                state = "ul"
            out.append(f"<li>{_inline(line[2:])}</li>")

        elif line == "":
            out.extend(_close_block(state))
            state = "none"

        else:
            # Paragraph text
            if state != "p":
                out.extend(_close_block(state))
                out.append("<p>")
                state = "p"
            out.append(_inline(line))

    out.extend(_close_block(state))
    return out


def _close_block(state: str) -> list[str]:
    """Return closing tag(s) for the current block state."""
    if state == "p":
        return ["</p>"]
    if state == "ul":
        return ["</ul>"]
    if state == "table_body":
        return ["</tbody></table>"]
    if state == "table_head":
        return ["</thead></table>"]
    return []


def _is_table_separator(line: str) -> bool:
    """Return True for lines like |---|---| or |:---|:---:|."""
    cells = [c.strip() for c in line.strip("|").split("|")]
    return bool(cells) and all(re.match(r"^:?-+:?$", c) for c in cells if c)


# ---------------------------------------------------------------------------
# Inline processing
# ---------------------------------------------------------------------------


def _inline(text: str) -> str:
    """Handle inline markdown within a single line of text.

    Processing order:
        1. Split on code spans ``...`` to protect them from further processing.
        2. HTML-escape non-code segments.
        3. Apply **bold** within non-code segments.

    Only handles constructs generated by the reporting layer.
    """
    # Split on backtick code spans: `...`
    segments = re.split(r"(`[^`]+`)", text)
    parts: list[str] = []
    for seg in segments:
        if len(seg) >= 2 and seg[0] == "`" and seg[-1] == "`":
            inner = _html_lib.escape(seg[1:-1])
            parts.append(f"<code>{inner}</code>")
        else:
            escaped = _html_lib.escape(seg)
            escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
            parts.append(escaped)
    return "".join(parts)


def _parse_image(line: str) -> str:
    """Convert ![alt](path) to an HTML <figure> block."""
    m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
    if not m:
        return f"<p>{_html_lib.escape(line)}</p>"
    alt = _html_lib.escape(m.group(1))
    src = m.group(2)   # path — not escaped; may contain valid path chars
    return (
        f'<figure>'
        f'<img src="{src}" alt="{alt}">'
        f'<figcaption>{alt}</figcaption>'
        f'</figure>'
    )


# ---------------------------------------------------------------------------
# HTML wrapper
# ---------------------------------------------------------------------------


def _wrap_html(body: str, title: str) -> str:
    """Wrap a body string in a minimal standalone HTML document."""
    escaped_title = _html_lib.escape(title)
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{escaped_title}</title>\n"
        f"<style>\n{_CSS}\n</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>"
    )
