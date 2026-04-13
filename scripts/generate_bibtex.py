"""Auto-generate .bib entries from a list of DOIs."""

import json
import sys
import urllib.request


def doi_to_bibtex(doi):
    """Fetch BibTeX entry for a DOI from CrossRef."""
    url = f"https://doi.org/{doi}"
    req = urllib.request.Request(url, headers={"Accept": "application/x-bibtex"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"Failed to fetch {doi}: {e}", file=sys.stderr)
        return None


def main():
    """Read DOIs from stdin (one per line) and output BibTeX."""
    dois = [line.strip() for line in sys.stdin if line.strip()]
    entries = []
    for doi in dois:
        bib = doi_to_bibtex(doi)
        if bib:
            entries.append(bib)
    print("\n\n".join(entries))


if __name__ == "__main__":
    main()
