# SuanLab

Academic research and education website at https://suanlab.com. Next.js App Router,
TypeScript, Tailwind and Markdown; exported to static HTML and deployed to GitHub
Pages. `WWW/` in the parent workspace is an archived site and is not maintained.

## Develop and verify

```bash
npm ci
npm run dev                 # http://localhost:3000
npm run check               # lint, content checks, regression tests, build, export checks
npm run typecheck           # also included after build in npm run check
npm start                   # preview out/ at http://127.0.0.1:4175 (built-in Node.js server)
```

The static build is in `out/`. `npm start` and `npm run preview` serve this export;
they do not use `next start`. Do not run `typecheck` concurrently with `next build`.
For the browser acceptance sweep, run `npm start`, then `npm run check:browser`.
This checks 35 routes in light/Korean and
dark/English modes at 390px, including WCAG A/AA rules and horizontal overflow.
It writes `.runtime/browser-check.json` and two screenshots. The current host uses
`/usr/bin/google-chrome`; override `SUANLAB_BROWSER_PATH` for another executable or
`SUANLAB_TEST_BASE` for an already-running preview. Browser checks are separate from
the portable CI gate because they require a browser and a preview server.
The sweep also checks blocked language storage, language switching, persisted and
shared builder drafts, personal prompts, workflow response handoff, comment draft
preservation, and the real visible comment frame in dark mode. The live comment
check requires a connection to Giscus.
Navigation regressions also cover system dark mode, desktop keyboard controls,
mobile menu closure, blog/QT filter history, and prompt search destinations.
An additional 96 layout cases cover 16 routes in both languages at 320, 768 and
1440px; results are in `.runtime/responsive-check.json`.
Inter font subsets are vendored with their license in `public/assets/fonts/inter/`;
builds and page views do not need a Google Fonts connection.
The preview serves public assets with CORS headers, matching GitHub Pages.
The browser sweep routes this export under its canonical public origin so external
widgets can load its styles without public-to-loopback network restrictions.
Production comments use an explicit dark canvas; HTTP previews use Giscus' built-in
theme. Comments initialize once after the theme resolves; later theme/language
changes preserve drafts. The embed waits until the browser mounts to keep streamed
article hydration consistent with the exported HTML.

## English localization

Keep curated display translations in `src/data/i18n/` using exact, whitespace-normalized
Korean source strings as keys. Wrap server page trees and client-rendered lists with
`LocalizedContent`; custom components should use `useDisplayText` or the existing
language provider for their own labels. Translate headings, descriptions and readable
attributes, while keeping IDs, URLs, filter values and citation exports canonical.

Original Korean articles, book prose and formal citations use `SourceText` or precise
`lang="ko"` regions. Show `OriginalLanguageNotice` when a page contains Korean source
prose. These markers describe source material; do not use them to suppress missing UI
translations. Presentation HTML is translated separately without changing code or math.

After building and starting the preview, run `npm run check:language` (also included in
`check:browser`). It scans every UI route family, sample original articles, video tabs
and every lecture slide, then checks locale switching and filter preservation. Reports
are written to `.runtime/language-check.json`; untranslated UI or document titles fail
the check. Add new translations alongside changes to their source text.

## Motion design

Shared motion lives in `src/components/motion/` and `src/styles/motion.css`.
Use `data-motion-reveal` for a one-time viewport entrance; shared cards already
include it. Content stays visible in exported HTML and without JavaScript.
Entrances preserve text opacity so labels keep their contrast while moving.
`MotionList` / `MotionItem` animate filtered results with lazy-loaded Motion layout
features. Keep stable item keys and preserve focus on the search field.

The hero and page-title pause controls share a site-wide preference with the footer.
The choice survives navigation/reload and also works when storage is denied.
OS reduced-motion settings take precedence. Decorative scenes pause outside the
viewport or when the tab is hidden; counters expose their final value to readers.
Buttons use 180ms, list movement 240ms and entrances 480ms; mobile entrances move
8px rather than 16px. The research page follows four scroll stages with normal
document scrolling. Publication timeline counts come from publication records;
map routes are schematic connections within selected trip records.

`npm run check:browser` includes `scripts/check-motion.cjs` for pause preferences,
reduced motion, research stages, timeline filters, map routes, denied storage,
no-JavaScript content, and runtime/hydration errors. Results and screenshots go to
`.runtime/motion-check.json` and `.runtime/motion-*.png`.
Accessibility sweeps await finite entrance animations and font loading before
measuring the settled layout; continuous decorative loops keep running.

## Learning content roadmap

[Learning content plan](docs/online-learning/README.md) prioritizes the design of the
new **Physics and Mathematics for AI** textbook. Its open-ended curriculum has no
week, semester, page or chapter limit: it connects mathematical and physical
foundations to AI mechanisms and research through a prerequisite graph. Detailed
knowledge domains, research paths and authoring standards accompany the reader
design, bilingual vector lesson and three initial Three.js scene specifications.

The first Colab course is [Python Programming](docs/online-learning/python-programming.md):
a nine-lesson plan and bilingual lesson 01 notebooks in `notebooks/python-programming/`.
Other Colab subjects remain a backlog. One hundred and three interactive textbook units are published at
[Online Book](https://suanlab.com/book/online/physics-math-for-ai/): logic and implication, quantifiers, sets and functions, relations and orders,
proof strategies, mathematical induction, fractions and units, real and complex numbers, equations and inequalities,
polynomial/rational functions, exponentials/logarithms, trigonometry/coordinates, composition/inverses, sequences/series,
asymptotic growth, counting and permutations, inclusion–exclusion, recurrences and generating functions,
graphs and DAGs, paths and connectivity, Boolean algebra, automata, lattices, finite-state transitions, vectors and forces, inner products and projection, linear transformations,
least squares, eigenvalues and PCA, gradients, the chain rule, Jacobians, backpropagation, SVD, bases/rank/nullspaces, determinants/traces, PSD matrices,
quadratic forms/Loewner order, Kronecker products, limits/continuity, differentiation/integration foundations, Taylor remainders,
series/uniform convergence, compactness, convexity, integral interchanges, implicit/inverse functions,
matrix derivatives, vector fields, geometric integrals, Gauss/Stokes and higher derivatives, complex polar maps, holomorphicity, contour integrals, residues,
singularities, analytic continuation, Hermitian products, Wirtinger gradients, phase interference, tensor shapes/contractions, multilinear maps, broadcasting, CP/Tucker/TT,
contraction networks, low-rank errors, symmetry and storage costs, groups/rings/fields, homomorphisms, quotients, finite fields, modules,
group actions, polynomial systems, invariants and equivariant representations, probability spaces/events, conditioning,
mass/density/CDFs, joint laws, Gaussian geometry, transformations, LLN/CLT, reparameterization, probability and moments, likelihood, Bayes,
information theory, ridge, SGD, differential equations, mechanics, statistical mechanics, Markov processes,
inverse problems and research evaluation.
Each has Korean/English theory, equations, numeric readouts and understanding checks.
Spatial labs use Three.js with a 2D fallback; calculus labs use interactive SVG computation graphs and derivative checks. Further labs use native SVG
distribution plots, exact/numerical trajectories and sensitivity sweeps with numeric evidence. Preview content lives in
`content/learning/physics-math-for-ai/`; typed metadata in `src/data/interactive-book.ts`.
The one hundred and three units include graduate-entry explanations, worked examples,
step-by-step concept diagrams, and three hundred and nine guided predict/try/explain experiments.
The [implementation sequence](docs/online-learning/physics-math-for-ai/implementation-sequence.md)
records prerequisite order, calculation contracts and the next authoring queue.
The book also exposes the complete bilingual scope of 51 domains and 509 topics, with related readings
and search. This scope is not a full-completion claim. Run `npm run learning:curriculum` after changing
the scope, its English translation or the coverage ledger to synchronize the static public map.
Run `npm run check:learning`, `npm run check:foundations`, `npm run check:logic` `npm run check:elementary` `npm run check:discrete` `npm run check:matrix-structures` `npm run check:analysis` `npm run check:complex` `npm run check:tensors` `npm run check:abstract-algebra` and `npm run check:probability` after building and starting the preview for bilingual
accessibility, responsive layout, simulation interactions and fallback checks.
See [implementation notes](docs/online-learning/physics-math-for-ai/preview-implementation.md)
for model conventions and scope. Notebook verification distinguishes local Jupyter from Colab.

## Where to edit

Blog filters use `?q=...&category=...&tag=...&page=2`; QT uses
`?book=요한복음&q=...&testament=new&tab=recent`. Reload and browser history restore
the selection. Prompt search opens `/prompts/?builder=rag-system` or
`/prompts/?snippet=snip-grounded-answer`; existing shared-input hashes still work.

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
| Prompt UI and workflow storage | `src/components/prompts/` |

Content is plain Markdown, not MDX. Blog files use `YYYYMMDD-slug.md` and YAML
metadata. QT filenames generate date-based slugs with suffixes for duplicate dates.
Use the content loaders for all route lists; do not independently recreate slugs.
After adding a QT file, run `npm run qt:register` and include
`src/config/qt-slugs.json` in the change. Existing and retired mappings remain
reserved, including when a new title sorts before an older duplicate-date entry.
Content loaders reuse parsed records within each build worker. Development checks
file stats and directory listings so edits, additions and deletions are visible.
Historical `Header.tsx`/`Footer.tsx` and `src/data/navigation.ts` are legacy;
the live shell uses `ModernHeader` and `ModernFooter`.

`src/data/editorial.ts` defines the homepage Top Conf venue selection, featured
active projects, stable research-to-publication/project/course IDs, and learning
paths. Homepage conference papers are grouped by venue and conference year, then
sorted by year/month; editorial venue order breaks date ties. All selected groups
appear in responsive columns sized to their content. Findings share the conference
card but keep their own label; journals and workshops are excluded. Counts and
paper links come from `src/data/publications/` automatically.
Research relationships indicate shared topics, not funding or authorship claims.
Blog reading suggestions are separately labeled keyword matches. Search covers
research outputs, projects, courses, online books, videos, posts and prompts.
Original content remains in its original language; the language selector translates
supported UI labels, not complete articles. The header measures translated menu
widths and switches to the navigation drawer when the search and utility buttons
need more space.

Historical QT variants formerly nested under `content/blog/content/qt/` are stored
in `archive/qt/`. All 878 source files are retained byte for byte and checked against
`archive/qt-manifest.json`. Of these, 876 filenames also exist in active QT: 196
have identical bytes and 680 contain differences. The two archive-only files remain
unpublished. Edit `content/qt/` for the live site; do not merge or delete historical
variants as automatic deduplication. Content validation rejects nested archives.
The preserved `AGENTS.md` files contain some historical file maps and command
descriptions; use this README and the current exports/types to locate active code.

Homepage identity meanings live in `src/data/lab-identity.ts`. Superintelligence,
Neural-networks and LAB remain visible; only the U/A terms type and erase. Their
width follows the entered text, moving the following words without reserved gaps. Site
branding metadata is centralized in `src/config/site.ts`; publication and course
titles retain their original terminology.

Title backgrounds use `PageHeader/MenuDotField.tsx` and `menu-dots.css`, with
deterministic dot patterns selected by the top-level route. Each menu has its own
motion; nested pages inherit it. A multicolor particle layer fills the entire
header, with percentage positions and pixel-sized dots for responsive layouts.
Up to 160 neighbor connections track moving dots at up to 30 frames per second;
mobile headers show and update up to 64 edges. QT uses the same field in its existing header.
Animation pauses offscreen, in hidden tabs, through the pause control, and when
reduced motion is requested.

The shared header logo is `public/assets/images/logo.svg`, with vector
geometry and traced lettering preserving the original artwork and no embedded bitmap or font dependency.
The PNG remains available for RSS readers. The homepage research graphic in
`HomeResearchNetwork` connects all five nodes with ten connections and bidirectional signals;
its motion follows the hero pause and reduced-motion controls.

## Contact email drafts

The contact form prepares a Gmail web draft or opens the visitor's email app;
the visitor must send it there. Copy and manual-copy fallbacks support other mail
services. It does not claim delivery, store inquiries, or send email from Pages.
The recipient is the first address in `site.contact.emails`. Direct website
submission requires a separately configured mail service endpoint.

## Research prompts and conference calendar

`/prompts/` contains 28 builders, 59 library templates and eight workflows.
The research toolkit adds RAG, agents, evaluation, multimodal analysis, Text-to-SQL,
schema extraction and service operations. New templates support Korean and English
output. Builders compose text locally; visitors execute it in their preferred model
tool. Workflow steps save the actual model response and pass it to the next builder.
Drafts, favorites, personal prompts and workflow progress use this browser's
localStorage. Manual output edits persist until explicit regeneration. Shared URLs
contain form inputs; they do not include manually edited output or workflow responses.

Edit typed content in `src/data/prompts/`; current research additions and reference
links live in `advanced.ts`. Preserve stable IDs for saved items and workflow steps.
Set `updatedAt` when changing content. Do not hardcode obsolete conference author
kits, API versions, model prices or invented citations.

`/deadlines/` separates submissions from the full conference schedule, with search,
year/area filters, a 30-day view, saved venues, timezone conversion and ICS export.
Maintain `src/data/conferences/index.ts` against each track's official CFP. Record
`sourceUrl` and `verifiedAt` only after checking the source; older records retain
their previous verification date. Keep unknown dates `null` and provisional dates
`status: 'tentative'`. Do not insert guessed month-start dates. For conferences,
`endDate` is inclusive; for invited revisions, explicitly use `kind: 'milestone'`.
Use IANA timezones plus `deadlineTime` for Pacific deadlines so daylight saving
is handled correctly. A row can override `timezone`, including `null` when that
track's time is unpublished. No timezone means date-only, never implicit AoE.
VLDB monthly submissions are explicit cycles through March 2027.

Calendar exports include future confirmed dates in the selected view. Exact
deadlines export in UTC; unpublished times and conference ranges use all-day
events. Conflicting official schedules remain tentative and are excluded. Dates
are maintained editorially, not scraped at runtime; check the CFP before submission.

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
Copy `.env.example` to `.env.local` and configure the AI and Slack credentials
described there. The example selects review mode; the code retains automatic
publishing when that setting is absent. No Telegram service or web admin password
is configured by this project.

Remote downloads have a whole-response timeout and streaming size limit (PDFs:
60 seconds and 50 MiB; metadata: 2 MiB). Image generation has a 120-second timeout
and normalizes decoded images to the format advertised by their URL. Generated
posts use unique names; explicit output collisions fail instead of replacing an
existing article or image. PDF input without extractable text and AI output without
an article body fail before publication. These checks do not certify factual accuracy.

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

Pull requests run the same checks as local development, including an explicit
post-build type check. Pushes to `master` run
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
