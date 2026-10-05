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
server for a static export. For the browser acceptance sweep, serve `out/` on port
4175, then run `npm run check:browser`. This checks 35 routes in light/Korean and
dark/English modes at 390px, including WCAG A/AA rules and horizontal overflow.
It writes `.runtime/browser-check.json` and two screenshots. The current host uses
`/usr/bin/google-chrome`; override `SUANLAB_BROWSER_PATH` for another executable or
`SUANLAB_TEST_BASE` for an already-running preview. Browser checks are separate from
the portable CI gate because they require a browser and a preview server.

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

Homepage identity meanings live in `src/data/lab-identity.ts`. Superintelligence,
Neural-networks and LAB remain fixed; only the U/A terms type and erase. Site
branding metadata is centralized in `src/config/site.ts`; publication and course
titles retain their original terminology.

## Contact email drafts

The contact form prepares a Gmail web draft or opens the visitor's email app;
the visitor must send it there. Copy and manual-copy fallbacks support other mail
services. It does not claim delivery, store inquiries, or send email from Pages.
The recipient is the first address in `site.contact.emails`. Direct website
submission requires a separately configured mail service endpoint.

## Generated content provenance

New topic and paper posts contain `provenance`: AI-assisted generation time,
`review.status: pending`, and input-derived source metadata. Paper sources record
the arXiv identifier/canonical URL or PDF URL/hash, title and extracted authors.
Query-bearing PDF URLs retain a content hash rather than publishing access parameters.
The source block is independent of AI-generated prose and citations. Historical
posts without provenance show “Review not recorded”; they are not labeled reviewed.

After checking the source, citations and claims, an editor can change
`provenance.review` to `status: reviewed`, `reviewer: "Name"`, and
`reviewedAt: "YYYY-MM-DD"`. Preserve the original source and generation timestamp.
Automatic publication does not imply human review. `SUANLAB_PUBLISH_MODE=review`
keeps posts local; retries honor that mode too. These records are editorial
metadata, not authentication or an automated fact-check.

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
npm run bot:retry -- JOB_ID
```

While the bot is running, use `/suanblog retry JOB_ID` in an allowed Slack channel.
For an offline retry, stop `suanlab-slack.service` first, run the command above,
then restart the service. Both entry points acquire the same Linux `flock`; a
second writer is refused. Retry outcomes and attempt timestamps are persisted,
including interrupted attempts. A successful retry never hides a partial batch's
generation failure. `/suanblog-status` reports queue counts, review/push counts,
retry count, average completed-attempt duration, and recent stages.
Never commit `.env.local`, credentials, or `.runtime/`. Do not run the one-time
`extract-*.js` migration scripts again.

## CI and release

Pull requests run the same checks as local development. Pushes to `master` run
checks, upload `out/`, then deploy to GitHub Pages. Export validation confirms every
sitemap URL has an HTML page and canonical, and scans every exported HTML page for
broken internal links, assets and fragment targets (including numeric slide hashes).
External availability is not part of this offline gate. Verify the deployed home, search, lecture presentation,
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
