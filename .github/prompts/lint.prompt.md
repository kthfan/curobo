---
agent: 'agent'
description: 'Health-check the wiki for quality issues'
tools: ['search/workspace']
---
Perform a health check of the knowledge wiki. Look for:

- Contradictions between pages
- Stale claims that newer sources have superseded
- Orphan pages with no inbound links
- Important concepts mentioned but lacking their own page
- Missing cross-references
- Pages missing required YAML frontmatter

Report findings as a prioritized list and offer to fix each issue.
