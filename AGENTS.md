# SUANLAB-NEXT

**Generated:** 2026-02-11
**Commit:** 04e7431
**Branch:** master

## OVERVIEW

Next.js 14.2.35 App Router static site for SuanLab research lab. Static export to GitHub Pages. TypeScript strict, Tailwind CSS, Radix UI. AI-powered blog generation (OpenAI GPT-4o + Gemini). 1890 markdown content files.

## STRUCTURE

```
suanlab-next/
├── src/
│   ├── app/              # App Router pages — each dir = route
│   │   ├── blog/         # Blog list + [slug] dynamic posts
│   │   ├── qt/           # Quiet Time devotionals + [slug]
│   │   ├── research/     # Research areas + [slug] detail
│   │   ├── admin/        # Admin panel (WIP, no auth)
│   │   ├── feed.xml/     # RSS route handler
│   │   └── {section}/    # course, lecture, project, publication, book, youtube, suan
│   ├── components/       # React components
│   │   ├── layout/       # Header (ModernHeader), Footer (ModernFooter), PageHeader
│   │   ├── ui/           # Shadcn/Radix primitives (button, card, badge, sheet, tabs)
│   │   ├── admin/        # TopicForm, PaperUpload, PreviewPane
│   │   └── seo/          # JsonLd structured data
│   ├── data/             # Static TS data files (has own AGENTS.md)
│   ├── lib/              # Business logic (has own AGENTS.md)
│   ├── styles/           # CSS: globals, blog-prose, legacy/
│   └── types/            # NavItem, SocialLink, ContactInfo
├── content/              # Markdown content (1890 files)
│   ├── blog/             # 1006 posts — YYYYMMDD-slug.md
│   ├── qt/               # 876 devotionals — YYYY-MM-DD Title.md
│   └── books/            # 8 book outlines
├── scripts/              # Tooling (has own AGENTS.md)
├── public/assets/        # Static images, fonts
└── out/                  # Build output (gitignored)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new page | `src/app/{route}/page.tsx` | App Router convention |
| Add dynamic route | `src/app/{route}/[slug]/page.tsx` | Must export `generateStaticParams` |
| Modify navigation | `src/data/navigation.ts` | 8 main sections with children |
| Add UI component | `src/components/ui/` | Shadcn pattern: CVA + Radix + Tailwind |
| Blog processing | `src/lib/blog.ts` | gray-matter + remark/rehype pipeline |
| QT processing | `src/lib/qt.ts` | Korean Bible books, custom parsing |
| AI features | `src/lib/ai/` | OpenAI, Gemini, prompts, image gen |
| PDF processing | `src/lib/pdf/` | pdf-parse, arXiv integration |
| Generate blog post | `scripts/blog/generate.ts` | CLI: topic-based or paper-based |
| Global metadata/SEO | `src/app/layout.tsx` | JSON-LD, OG tags, GA tracking |
| Theme/dark mode | `src/components/theme-provider.tsx` | next-themes, class-based |
| RSS feed | `src/app/feed.xml/route.ts` | GET handler, 50 recent posts |
| Sitemap | `src/app/sitemap.ts` | Dynamic generation |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `getAllPosts` | fn | lib/blog.ts | Returns all blog posts sorted by date |
| `getPostBySlugWithHtml` | fn | lib/blog.ts | Markdown → HTML for single post |
| `getAllQTEntries` | fn | lib/qt.ts | Returns all QT entries |
| `getQTByBibleBook` | fn | lib/qt.ts | Groups QT by Bible book |
| `generateWithOpenAI` | fn | lib/ai/claude.ts | GPT-4o text generation |
| `generateWithGemini` | fn | lib/ai/claude.ts | Gemini generation (fallback) |
| `generateWithDualAI` | fn | lib/ai/claude.ts | OpenAI → Gemini fallback strategy |
| `parsePdfFromUrl` | fn | lib/pdf/parser.ts | PDF text extraction |
| `fetchArxivMetadata` | fn | lib/pdf/arxiv.ts | arXiv paper metadata |
| `cn` | fn | lib/utils.ts | Tailwind class merging (clsx + twMerge) |
| `navigation` | const | data/navigation.ts | Site navigation structure |
| `BIBLE_BOOKS` | const | lib/qt.ts | 66 Bible books with Korean names |

## CONVENTIONS

- **Imports**: Use `@/*` path alias (maps to `./src/*`)
- **Static export**: `output: 'export'` — no server-side features (no API routes except feed.xml, no middleware)
- **Trailing slashes**: Enabled (`trailingSlash: true`) for GitHub Pages
- **Images**: `unoptimized: true` — no Next.js Image Optimization
- **Components**: Shadcn UI pattern — copy-paste Radix primitives into `components/ui/`
- **Dark mode**: Class-based via `next-themes` + Tailwind `darkMode: ["class"]`
- **Colors**: HSL CSS variables (--primary, --secondary, etc.) consumed by Tailwind
- **Content files**: Pure markdown + gray-matter frontmatter (NOT MDX)
- **Reading time**: English 200 WPM, Korean 500 chars/min
- **Blog filenames**: `YYYYMMDD-slug.md`
- **QT filenames**: `YYYY-MM-DD Title.md` (Korean titles)
- **Data files**: TypeScript exports in `src/data/` — no database, no CMS
- **No tests**: No test framework configured
- **No auth**: Admin pages exist but are unprotected

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT** add API routes (except route handlers for static gen) — static export only
- **DO NOT** use `next/image` optimization features — `unoptimized: true` is required
- **DO NOT** use server-side features (getServerSideProps, middleware, cookies) — static export
- **DO NOT** modify `src/styles/legacy/` — ported from WWW, marked "DO NOT CHANGE"
- **DO NOT** use MDX — content uses plain markdown + gray-matter
- **DO NOT** commit `.env.local` — has API keys

## COMMANDS

```bash
npm run dev              # Dev server (localhost:3000)
npm run build            # Static export to ./out
npm run lint             # ESLint (next/core-web-vitals + next/typescript)
npm run blog:generate    # Interactive AI blog generation
npm run blog:topic       # Generate from topic/keyword
npm run blog:paper       # Generate from arXiv/PDF paper
npm run bot:telegram     # Start Telegram bot
npm run bot:slack        # Start Slack bot
```

## NOTES

- Deployment: GitHub Actions → `npm run build` → upload `./out` → GitHub Pages
- CI triggers on push to `master` or manual dispatch
- BASE_URL: `https://suanlab.com`
- GA: `G-PYEC6PCW0P`
- Blog is mostly paper reviews (auto-generated from arXiv via AI pipeline)
- QT content = "Quiet Time" Christian devotionals with Korean Bible references
- Some directories have restricted permissions (700) — WIP or sensitive content
- Largest source files: publications/index.ts (2672 lines), youtube/index.ts (1311 lines) — data arrays
