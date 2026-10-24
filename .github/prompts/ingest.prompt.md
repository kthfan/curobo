---
agent: 'agent'
description: 'Ingest a new source document into the wiki'
tools: ['search/workspace', 'vscode/editFiles']
---
Ingest the provided source document into the knowledge wiki.

1. Read the raw source document thoroughly.
2. Discuss the 3-5 key takeaways with me before proceeding.
3. Create a source summary page in wiki/sources/.
4. Update wiki/index.md with the new entry.
5. Update or create relevant entity pages in wiki/entities/.
6. Update or create relevant concept pages in wiki/concepts/.
7. Check existing wiki pages for contradictions or updates needed.
8. Append an entry to wiki/log.md.

Follow all conventions in [copilot-instructions.md](../../.github/copilot-instructions.md).
