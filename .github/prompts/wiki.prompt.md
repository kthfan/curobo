# Wiki Maintenance Instructions

## Wiki Structure
This repository contains a knowledge wiki maintained by AI.
- `curobo/` contains immutable source documents. Never modify files here.
- `wiki/` contains AI-generated markdown pages. All wiki content goes here.
- `wiki/index.md` catalogs every wiki page with a link and one-line summary.
- `wiki/log.md` is an append-only chronological record of operations.

## Page Conventions
- Use YAML frontmatter with tags, date_created, date_updated, and source_count
- Use [[wiki-links]] for cross-references between wiki pages
- Every claim should cite its source using [Source Title](../sources/source-name.md)
- Flag contradictions explicitly with a > [!WARNING] callout
- Use Mermaid diagrams for process flows and architecture

## Ingest Workflow
When processing a new source:
1. Read the source document completely
2. Create or update a summary page in wiki/sources/
3. Update wiki/index.md with the new page
4. Update all relevant entity and concept pages
5. Check existing wiki pages for contradictions or updates needed
6. Append an entry to wiki/log.md with format:
   ## [YYYY-MM-DD] ingest | Source Title

## Quality Rules
- Never fabricate information not in the sources
- Preserve nuance; do not oversimplify policy positions
- Note when information may be outdated
- Track which sources support each claim
