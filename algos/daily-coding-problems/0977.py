"""
Implement a URL shortener with the following methods:

- shorten(url), which shortens the url into a six-character alphanumeric string, such
  as zLg6wl.

- restore(short), which expands the shortened string into the original url. If no
  such shortened string exists, return null.

Hint: What if we enter the same URL twice?
"""

import hashlib


class UrlShortener:
    def __init__(self):
        self._d = {}

    def shorten(self, url: str) -> str:
        k = self._hash(url)
        k = self._add(k, url)
        return k

    def _hash(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:6]

    def _add(self, k, url) -> str:
        while self._d.get(k) is not None:
            k = self._hash(k)
        self._d[k] = url
        return k

    def restore(self, k: str) -> str | None:
        return self._d.get(k)


shortener = UrlShortener()
assert shortener.restore(shortener.shorten("google.com")) == "google.com"
assert shortener.shorten("google.co.uk") != shortener.shorten("google.co.uk")
