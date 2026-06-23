#!/usr/bin/env python3
"""Fetch LMI help-center article URLs into data/lmi/urls_kb.txt."""
import scripts._bootstrap  # noqa: F401

import requests

urls = []
page = "https://support.lmi3d.com/api/v2/help_center/en-us/articles.json?per_page=100"
while page:
    data = requests.get(page, timeout=30).json()
    urls += [article["html_url"] for article in data["articles"]]
    page = data.get("next_page")

output = __import__("pathlib").Path("data/lmi/urls_kb.txt")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(urls) + "\n", encoding="utf-8")
print(f"{len(urls)} article URLs written to {output}")
