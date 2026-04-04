# APP — Route Layer

## OVERVIEW

Next.js App Router pages, layouts, metadata, and special route files. This layer assembles `src/data/`, `src/lib/`, and `src/components/` into static pages.

## STRUCTURE

```
app/
├── layout.tsx            # Global metadata, ThemeProvider, header/footer, GA, JSON-LD
├── page.tsx              # Homepage; mixes static data with recent blog posts
├── {section}/page.tsx    # Section landing pages (course, project, publication, youtube, etc.)
├── {section}/layout.tsx  # Section wrappers when route-specific layout or metadata differs
├── blog/[slug]/page.tsx  # Markdown-driven detail route
├── qt/[slug]/page.tsx    # QT detail route with generated slugs
├── admin/                # Client-gated admin area
└── {feed.xml,sitemap.ts,robots.ts}  # Special static-export files
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add top-level page | `page.tsx` in route dir | Pair with `metadata` when page is indexable |
| Add dynamic page | `{route}/[slug]/page.tsx` | Must export `generateStaticParams()` |
| Change global shell | `layout.tsx` | Header, footer, ThemeProvider, GA, JSON-LD |
| Change sitemap behavior | `sitemap.ts` | Reads `content/blog` and `content/qt` directly |
| Change robots rules | `robots.ts` | Currently disallows `/admin/` and `/api/` |
| Change RSS feed | `feed.xml/route.ts` | Only route handler in this tree |
| Touch admin pages | `admin/` | Client-side gate only; not secure auth |

## CONVENTIONS

- Dynamic detail routes require `generateStaticParams()` for static export; params may be sync or async depending on the file pattern already in use
- Page metadata lives beside the route; detail pages often add `generateMetadata()`
- `PageHeader` is the standard section header for non-home pages
- Pages compose prebuilt data/loaders; route files should stay thin when logic belongs in `src/lib/`
- `sitemap.ts` and `feed.xml/route.ts` are allowed build-time exceptions that read filesystem-backed content
- `search/page.tsx` preloads searchable data at build time, then hands off to a client component
- Admin uses `AdminAuthGate` with localStorage plus a hardcoded password; document this as a weak gate, not real protection

## ANTI-PATTERNS

- **DO NOT** add middleware, server actions that require runtime state, or request-only APIs here — static export forbids them
- **DO NOT** add a new dynamic route without `generateStaticParams()`
- **DO NOT** move markdown parsing into route files — keep that in `src/lib/`
- **DO NOT** treat `/admin/` as secure because it renders behind a client-side password prompt
- **DO NOT** add extra route handlers casually — `feed.xml/route.ts` is the explicit exception
