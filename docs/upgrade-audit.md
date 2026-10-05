# Website upgrade completion audit

Updated: 2026-10-05. This audit retains the original full upgrade scope; a deployed
increment does not imply the entire plan is complete.

| Scope | Implemented | Remaining acceptance work |
| --- | --- | --- |
| Repository and reference inventory | Active Next.js/static export, archived WWW, source ZIP reviewed; existing AGENTS preserved | Public landing and 13-slide presentation verified in browser; authenticated actions not exercised |
| Source of truth and content quality | Central URL/navigation; loader-based sitemap; IDs, metadata and editorial reference checks; full internal route/asset scan | Add explicit content provenance and review state to new generated reviews; broader duplicate/date auditing |
| Research and home | Refined home, ordered representative publications with reasons, active project cards; stable topic connections between 7 research areas and publications/projects/courses | Structured project problem/approach/outcomes with evidence, reciprocal links; do not invent performance results |
| Education | Three prerequisite/level-based paths through lectures/videos/book outlines; restored 33 original PDFs; repaired 3 old links | Expand authored lessons as teaching material becomes available |
| Common UI | Shared shell, search and filters, responsive/dark controls, profile contents and removal of unsupported skill percentages | Broader accessibility and bilingual-control audit across all long pages |
| Slack operations | Enabled systemd service, serialized durable queue, deduplication, restart recovery, saved-file publish retry, review mode, safe publication, metadata/asset checks | Persist retry outcomes, operational metrics, finer failure stages; evaluate isolated worktree to separate publishing from development |
| Presentation pilot | One Markdown source produces reading and 10-slide AI lesson, code/math/figure/references, notes, hash navigation, fullscreen and PDF | Verified locally; repeat key smoke checks after deployment |
| Delivery | PR/push CI gates lint, content validation, regression tests, static build and export checks | Monitor this increment's deployment and verify live pages |

## Verification boundaries

- Local browser checks cover 360–1440px home layouts, dark mode, language, navigation,
  search-to-publication links, 390px pilot slides, reading, keyboard and fullscreen.
- The pilot exports a 10-page landscape PDF. Python lesson assertions pass.
- Unit/regression tests use disposable Git repositories; no unsolicited Slack
  message or paid AI generation was sent during verification.
- Internal HTML links and assets are checked. External sites, content factuality,
  and every deep-link fragment require separate evidence.
- Legacy PDFs were copied from `../WWW/`; originals were not changed.
- Historical generated posts may contain unsupported references. New validation
  checks structure and files, not factual accuracy; provenance/review remains open.
