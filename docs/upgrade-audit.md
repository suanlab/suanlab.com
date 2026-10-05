# Website upgrade completion audit

Updated: 2026-10-05. This checklist covers the original upgrade plan across the
existing static website, teaching materials, and Slack publishing service.

| Scope | Implementation and acceptance evidence |
| --- | --- |
| Repository and reference inventory | Active Next.js/static export and source ZIP reviewed; public reference landing and 13-slide presentation inspected in a browser. See `reference-analysis.md`. Existing AGENTS files and archived WWW remain unchanged. |
| Source of truth and content quality | Central site URL/navigation, loader-based sitemap, editorial ID checks, duplicate body/publication checks, calendar-date validation, and exported link/asset/fragment/canonical checks. Date-only and date-range QT filenames now retain valid dates and titles without changing URLs. |
| Research and homepage | Refined existing design, ordered representative publications with selection reasons, current project cards, connections across seven research areas, and reciprocal project/course links. Selected projects describe problem, approach, and evidence availability without inventing performance results. |
| Education | Three level/prerequisite-based learning paths, 33 restored original PDFs, and three repaired book links. One complete AI lesson shares Markdown between reading and ten slides, including code, mathematics, a figure, references, speaker notes, keyboard/hash/fullscreen navigation, and a ten-page print PDF. |
| Common UI | Responsive/dark layouts, Korean/English interface controls, long-page contents, keyboard filters, accessible names, readable contrast, and contained code/math/table scrolling. Unsupported profile skill percentages removed. |
| Content provenance | Newly generated posts record input-derived source metadata, generation date, and pending review. Reviewed status requires reviewer/date evidence. Historical posts explicitly show that review was not recorded. Private or query-bearing PDF URLs are suppressed; source hashes remain available. |
| Slack operations | Active systemd service; durable serialized queue, deduplication, process lock, restart recovery, detailed phases, retry attempt history, metrics, saved-file publication retry, and review/automatic publication modes. Git publication guards reject unrelated staged files, behind branches, and unrelated unpushed commit history. |
| Delivery | PR/push CI gates lint, content validation, regression tests, static build, and export checks. Local acceptance passed; the final release is verified through GitHub Actions and production smoke checks after this commit. |

## Verification

- `npm run check`: lint, content validation (1,230 sitemap URLs), 15 regression
  tests, build, and 1,233 exported HTML pages passed. `/search/` is intentionally
  noindex and excluded from the sitemap.
- `npm run typecheck` passed. Tests include isolated Git repositories, persisted
  queue recovery/retries, cross-process locks, YAML provenance, and legacy dates.
- `npm run check:browser`: 35 representative routes in light/Korean and dark/English
  modes (70 cases) passed automated accessibility and horizontal-overflow checks.
  The ignored `.runtime/browser-check.json` records results.
- Additional browser checks covered 360–1,440px article layouts, keyboard tag
  filtering, reciprocal research links, bibliography copy feedback, and the AI
  lesson's navigation, notes, mathematics, and ten-page dark-theme print output.
- Slack service startup and read-only authentication were verified. No unsolicited
  Slack messages or paid generation were used as acceptance tests.

## Operational decisions and limits

A separate publishing worktree was evaluated. The current checkout is retained
with a kernel-enforced single writer and branch, staging, remote-history, and file
scope checks. Operators can retry saved output through `/suanblog retry JOB_ID`;
offline retry requires stopping the service first. See `../scripts/systemd/README.md`.

Provenance and structural validation do not establish factual accuracy. Historical
articles are not retrospectively certified, and unpublished project metrics are
not inferred. The browser sweep covers representative templates and controls,
not every content page or interaction. Reference-site authenticated actions and
remote presentation control were inspected in source, not exercised live.
Additional authored lessons and historical editorial review are ongoing content
maintenance beyond the completed one-lesson pilot.
