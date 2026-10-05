# SuanLab

Academic research and education website at https://suanlab.com. Next.js App Router,
TypeScript, Tailwind and Markdown; exported to static HTML and deployed to GitHub
Pages. `WWW/` in the parent workspace is an archived site and is not maintained.

## Develop and verify

```bash
npm ci
npm run dev                 # http://localhost:3000
npm run check               # lint, content checks, regression tests, build, export checks
npm run typecheck           # run after build; do not run concurrently with next build
```

The static build is in `out/`. Preview with
`python3 -m http.server 4173 --directory out`. `next start` is not the preview
server for a static export.

## Where to edit

| Area | Source |
| --- | --- |
| Site URL and contact details | `src/config/site.ts` |
| Active header navigation | `src/data/site-navigation.ts` |
| Research, publications, projects, lectures, videos | `src/data/` |
| Blog, QT, books | `content/` |
| Home and shared components | `src/components/` |
| Route assembly and sitemap | `src/app/` |
| Markdown and presentation helpers | `src/lib/` |
| UI translations | `src/components/language-provider.tsx` |

Content is plain Markdown, not MDX. Blog files use `YYYYMMDD-slug.md` and YAML
metadata. QT filenames generate date-based slugs with suffixes for duplicate dates.
Use the content loaders for all route lists; do not independently recreate slugs.
Historical `Header.tsx`/`Footer.tsx` and `src/data/navigation.ts` are legacy;
the live shell uses `ModernHeader` and `ModernFooter`.

`src/data/editorial.ts` defines the ordered featured publications with selection
reasons, featured active projects, stable research-to-publication/project/course IDs,
and learning paths. Update it deliberately rather than relying on array order.
Research relationships indicate shared topics, not funding or authorship claims.
Blog reading suggestions are separately labeled keyword matches. Search covers
research outputs, projects, courses, online books, videos, posts and prompts.
Original content remains in its original language; the language selector translates
supported UI labels, not complete articles.

## Lecture presentations

`/lecture/{slug}/present/` uses the same lecture title, overview and topics as the
reading page. Courses without authored material have overview decks. The AI pilot has a complete introductory lesson with code, equations, a diagram, exercises, and references. Use arrow
keys, Home/End, the slide selector, full screen, presenter notes, or Print/PDF.
`#2` links directly to slide 2. For authored lessons, edit `content/lectures/{slug}.md`: YAML `title`, `lecture`,
and `date`, then slides separated by top-level `---`. Each slide begins with `#`;
optional `<!-- notes: ... -->` blocks appear only in presenter notes. Fenced code,
math, images and tables are rendered at build time. `/lecture/{slug}/notes/` and
the presentation share this exact source; the reading view includes a contents
list and direct links to corresponding slides. Other overview decks use
`src/data/lectures/` as their shared source.
Presentation controls remain local: no database, remote control or live audience
synchronization is required.

## Slack operations

See [service installation and operations](scripts/systemd/README.md).
The bot is a separate long-running process, not part of GitHub Pages.

Requests run serially. `.runtime/slack-jobs.json` stores status and saved files;
completed/active requests are deduplicated, including arXiv URL/ID variants.
After an interrupted job, inspect saved files before retrying. Jobs are not
silently regenerated on restart. `/suanblog-status` includes recent job states.

`SUANLAB_PUBLISH_MODE=review` saves generated files for review instead of pushing.
Default behavior retains automatic publication. Publication validates required metadata,
URL schemes and local images before staging blog files; new inline blog images are
included in the same commit. For a job that already saved
files, retry publication without paying for generation again:

```bash
npx tsx scripts/blog/retry-publish.ts JOB_ID
```

Do not run that command while a generation/publish job is active. The running bot
owns its ledger; a manual publish does not rewrite historical job status.
Never commit `.env.local`, credentials, or `.runtime/`. Do not run the one-time
`extract-*.js` migration scripts again.

## CI and release

Pull requests run the same checks as local development. Pushes to `master` run
checks, upload `out/`, then deploy to GitHub Pages. Export validation confirms every
sitemap URL has an HTML page and canonical, and scans every exported HTML page for
broken internal links and assets (external availability and fragment targets are not checked). Verify the deployed home, search, lecture presentation,
and sitemap after release. New dynamic routes must define `generateStaticParams`.

Use imperative commit subjects describing observable changes. Include affected
pages, verification results and screenshots for UI changes in pull requests.
Keep generated assets and content changes reviewable; do not bundle secrets or
unrelated work into publication commits.

## Upgrade tracking

See [upgrade completion audit](docs/upgrade-audit.md) for completed work, remaining
scope and verification limits. Existing `AGENTS.md` files have been preserved.

## Reference implementation reviewed

See the [source and live-page analysis](docs/reference-analysis.md) for evidence and limitations.

The supplied `idc-audit-ontology-source.zip` documents Claude Code as its development
tool (README section 10). Its application LLM is separately configurable. It uses
Next.js/React, PostgreSQL/Drizzle, RDF/SPARQL and a Go host agent. Its public
`/present` route is separate from authenticated application routes, with custom
React slides, keyboard/build-step navigation, touch, URL hashes, full screen,
3D scenes, and an API-backed live remote-control channel. The source findings were cross-checked against the public landing page and 13-slide
presentation on 2026-10-05: both returned HTTP 200 in a real browser without login.
The landing page explains observation → investigation → registration → decision →
remediation → recheck → evidence. Authenticated workflows, live remote control,
and actual model calls were not exercised. SuanLab
adopts the reusable-content and direct-slide-link patterns, with an independent
implementation suited to static hosting; it does not copy the server-backed control
channel or the security-auditing application.
