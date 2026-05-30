"""Tests for src/reporting/html.py."""

from __future__ import annotations

from src.reporting.html import _inline, _is_table_separator, markdown_to_html

# ---------------------------------------------------------------------------
# markdown_to_html — top-level
# ---------------------------------------------------------------------------


def test_returns_string() -> None:
    assert isinstance(markdown_to_html("hello"), str)


def test_empty_input_returns_valid_html() -> None:
    out = markdown_to_html("")
    assert "<!DOCTYPE html>" in out
    assert "<body>" in out


def test_title_appears_in_title_tag() -> None:
    out = markdown_to_html("", title="My Report")
    assert "<title>My Report</title>" in out


def test_title_escaped() -> None:
    out = markdown_to_html("", title="R&D <report>")
    assert "R&amp;D" in out
    assert "&lt;report&gt;" in out


def test_has_doctype() -> None:
    assert markdown_to_html("").startswith("<!DOCTYPE html>")


def test_has_body_tags() -> None:
    out = markdown_to_html("some text")
    assert "<body>" in out
    assert "</body>" in out


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------


def test_h1_rendered() -> None:
    out = markdown_to_html("# My Title")
    assert "<h1>My Title</h1>" in out


def test_h2_rendered() -> None:
    out = markdown_to_html("## Section")
    assert "<h2>Section</h2>" in out


def test_h3_rendered() -> None:
    out = markdown_to_html("### Subsection")
    assert "<h3>Subsection</h3>" in out


def test_heading_text_escaped() -> None:
    out = markdown_to_html("# Title <with> &amp; chars")
    assert "&lt;with&gt;" in out


# ---------------------------------------------------------------------------
# Horizontal rule
# ---------------------------------------------------------------------------


def test_hr_rendered() -> None:
    out = markdown_to_html("---")
    assert "<hr>" in out


# ---------------------------------------------------------------------------
# Pipe tables
# ---------------------------------------------------------------------------


def test_table_has_table_tag() -> None:
    md = "| A | B |\n|---|---|\n| x | y |"
    out = markdown_to_html(md)
    assert "<table>" in out
    assert "</table>" in out


def test_table_has_thead_tbody() -> None:
    md = "| A | B |\n|---|---|\n| x | y |"
    out = markdown_to_html(md)
    assert "<thead>" in out
    assert "<tbody>" in out


def test_table_header_cells_are_th() -> None:
    md = "| Field | Value |\n|---|---|\n| k | v |"
    out = markdown_to_html(md)
    assert "<th>Field</th>" in out
    assert "<th>Value</th>" in out


def test_table_data_cells_are_td() -> None:
    md = "| Field | Value |\n|---|---|\n| k | v |"
    out = markdown_to_html(md)
    assert "<td>k</td>" in out
    assert "<td>v</td>" in out


def test_table_separator_not_rendered() -> None:
    md = "| A |\n|---|\n| x |"
    out = markdown_to_html(md)
    assert "|---|" not in out
    assert "---" not in out or "<hr>" in out  # only hr from literal ---


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def test_image_rendered_as_figure() -> None:
    out = markdown_to_html("![My Chart](path/to/chart.png)")
    assert "<figure>" in out
    assert "<img" in out
    assert 'src="path/to/chart.png"' in out


def test_image_alt_text_escaped() -> None:
    out = markdown_to_html("![R&D <plot>](plot.png)")
    assert "R&amp;D" in out
    assert "&lt;plot&gt;" in out


def test_image_has_figcaption() -> None:
    out = markdown_to_html("![My Chart](plot.png)")
    assert "<figcaption>My Chart</figcaption>" in out


# ---------------------------------------------------------------------------
# Inline code spans
# ---------------------------------------------------------------------------


def test_inline_code_span() -> None:
    out = markdown_to_html("Use `config.json` here.")
    assert "<code>config.json</code>" in out


def test_inline_code_escapes_content() -> None:
    out = markdown_to_html("Use `a < b`.")
    assert "<code>a &lt; b</code>" in out


# ---------------------------------------------------------------------------
# Bold
# ---------------------------------------------------------------------------


def test_bold_rendered() -> None:
    out = markdown_to_html("This is **important**.")
    assert "<strong>important</strong>" in out


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


def test_ul_rendered() -> None:
    out = markdown_to_html("* item one\n* item two")
    assert "<ul>" in out
    assert "<li>item one</li>" in out
    assert "<li>item two</li>" in out


def test_dash_list_rendered() -> None:
    out = markdown_to_html("- item")
    assert "<li>item</li>" in out


def test_ul_closed_on_blank_line() -> None:
    out = markdown_to_html("* item\n\nNext paragraph")
    assert "</ul>" in out


# ---------------------------------------------------------------------------
# Paragraphs
# ---------------------------------------------------------------------------


def test_plain_text_wrapped_in_p() -> None:
    out = markdown_to_html("Just some text here.")
    assert "<p>" in out
    assert "Just some text here." in out


def test_paragraph_closed_on_blank_line() -> None:
    out = markdown_to_html("Para one.\n\nPara two.")
    assert out.count("<p>") >= 2
    assert out.count("</p>") >= 2


# ---------------------------------------------------------------------------
# HTML special character escaping in text
# ---------------------------------------------------------------------------


def test_ampersand_escaped_in_text() -> None:
    out = markdown_to_html("A & B")
    assert "&amp;" in out
    assert "A & B" not in out


def test_lt_escaped_in_text() -> None:
    out = markdown_to_html("a < b")
    assert "&lt;" in out


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_output() -> None:
    md = "# Title\n\n## Section\n\n| A | B |\n|---|---|\n| x | y |"
    assert markdown_to_html(md) == markdown_to_html(md)


# ---------------------------------------------------------------------------
# Round-trip: full reporting markdown renders to valid HTML
# ---------------------------------------------------------------------------


def test_full_report_round_trip() -> None:
    md = (
        "# Experiment Report: test_exp\n\n"
        "## Metadata\n\n"
        "| Field | Value |\n"
        "|---|---|\n"
        "| Experiment | `test_exp` |\n"
        "| Strategy | `EqualWeight(freq=ME)` |\n\n"
        "## Performance Metrics\n\n"
        "| Metric | Value |\n"
        "|---|---|\n"
        "| Sharpe Ratio | 0.6500 |\n\n"
        "---\n\n"
        "Report version: 1\n"
        "Generated: 2026-05-23T00:00:00+00:00\n"
        "Source experiment: test_exp\n"
    )
    out = markdown_to_html(md, title="Test Report")
    assert "<h1>Experiment Report: test_exp</h1>" in out
    assert "<h2>Metadata</h2>" in out
    assert "<h2>Performance Metrics</h2>" in out
    assert "<code>test_exp</code>" in out
    assert "0.6500" in out
    assert "<hr>" in out


# ---------------------------------------------------------------------------
# _inline helper
# ---------------------------------------------------------------------------


def test_inline_plain_text_escaped() -> None:
    assert _inline("a < b") == "a &lt; b"


def test_inline_code_span() -> None:
    assert _inline("`foo`") == "<code>foo</code>"


def test_inline_bold() -> None:
    assert _inline("**bold**") == "<strong>bold</strong>"


def test_inline_mixed() -> None:
    result = _inline("`code` and **bold**")
    assert "<code>code</code>" in result
    assert "<strong>bold</strong>" in result


# ---------------------------------------------------------------------------
# _is_table_separator
# ---------------------------------------------------------------------------


def test_separator_simple() -> None:
    assert _is_table_separator("|---|---|")


def test_separator_with_spaces() -> None:
    assert _is_table_separator("| --- | --- |")


def test_separator_aligned() -> None:
    assert _is_table_separator("|:---|:---:|")


def test_non_separator() -> None:
    assert not _is_table_separator("| data | row |")
