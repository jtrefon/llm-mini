# Release v1.0.0 — Public release

This release marks the public migration of the project to GitHub: https://github.com/jtrefon/llm-mini

Highlights
- Project made public and published on GitHub
- Small docs and contributing updates

How to get the release
- Tag: `v1.0.0`

Notes
- If you want a GitHub Release with release notes and binary assets, run:

  gh auth login
  gh release create v1.0.0 --title "Public release v1.0.0" --notes-file docs/RELEASE.md

- To add topics and update the repository description via the GitHub CLI:

  gh auth login
  gh repo edit jtrefon/llm-mini --description "Educational tiny transformer LLM from scratch" --add-topic tiny-transformer llm educational python

Thank you to everyone who contributed.