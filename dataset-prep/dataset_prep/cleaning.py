"""HTML → Markdown conversion with fenced code blocks.

StackOverflow stores posts as HTML. ``<pre><code>`` becomes a fenced block;
a normalized language tag is included only when the source CSS classes provide
a recognized hint. This conversion preserves structure imperfectly and does
not itself guarantee downstream training quality.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString
from markdownify import MarkdownConverter

LANG_HINT_RE = re.compile(r"\blang(?:uage)?-([a-z0-9+#-]+)\b", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\n{3,}")
TRAILING_WS_RE = re.compile(r"[ \t]+\n")


class _SOConverter(MarkdownConverter):
    """Markdownify converter tuned for StackOverflow HTML.

    - Forces ATX headings
    - Emits fenced code blocks with a language hint when SO provides one
      (CSS class like `lang-python` on <pre> or <code>)
    - Strips noisy attributes
    """

    def convert_pre(self, el, text, parent_tags):  # noqa: ARG002
        if not text:
            return ""
        lang = _detect_language(el)
        body = text.strip("\n")
        fence = "```"
        return f"\n\n{fence}{lang}\n{body}\n{fence}\n\n"


def _detect_language(pre_tag) -> str:
    candidates: list[str] = []
    for tag in (pre_tag, *pre_tag.find_all("code", limit=1)):
        for cls in tag.get("class", []) or []:
            if match := LANG_HINT_RE.search(cls):
                candidates.append(match.group(1).lower())
    if not candidates:
        return ""
    lang = candidates[0]
    aliases = {"py": "python", "js": "javascript", "sh": "bash"}
    return aliases.get(lang, lang)


def html_to_markdown(html: str) -> str:
    """Convert a StackOverflow HTML post body into clean markdown.

    Empty/None input returns an empty string.
    """
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    for tag in soup.find_all(True):
        if isinstance(tag, NavigableString):
            continue
        for attr in list(tag.attrs):
            if attr not in {"class", "href", "src"}:
                del tag.attrs[attr]

    converter = _SOConverter(
        heading_style="ATX",
        bullets="-",
        code_language="",
        strip=["img"],
    )
    md = converter.convert_soup(soup)

    md = TRAILING_WS_RE.sub("\n", md)
    md = WHITESPACE_RE.sub("\n\n", md)
    return md.strip()
