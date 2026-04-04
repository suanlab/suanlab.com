# CONTENT — Markdown Source Files

## OVERVIEW

Source markdown corpus for the static site. Three content families live here: blog posts, QT devotionals, and books, each with different filename and metadata rules.

## STRUCTURE

```
content/
├── blog/    # Blog/tutorial posts; full frontmatter; rendered by `src/lib/blog.ts`
├── qt/      # Quiet Time devotionals; empty frontmatter; filename-driven metadata via `src/lib/qt.ts`
└── books/   # Book outlines; frontmatter plus long-form markdown rendered by `src/lib/books.ts`
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add or edit blog post | `blog/*.md` | Filename `YYYYMMDD-slug.md` |
| Add or edit QT entry | `qt/*.md` | Usually `YYYY-MM-DD Title.md`; date-only legacy files exist |
| Add or edit book outline | `books/*.md` | Slug comes from filename |
| Change rendering rules | `../src/lib/blog.ts`, `../src/lib/qt.ts`, or `../src/lib/books.ts` | Content behavior is code-driven |
| Generate blog posts | `../scripts/blog/` | Preferred path for AI-generated posts |

## CONVENTIONS

- Blog posts use YAML frontmatter with `title`, `date`, `excerpt`, `category`, `tags`, and optional `thumbnail`
- QT files intentionally keep empty frontmatter (`---` / `---`); title and date come from the filename, Bible references from the body, and duplicate dates may get suffixed slugs
- Book files use frontmatter with `title`, `subtitle`, `author`, `date`, and `image`
- Blog markdown may include tables, fenced code blocks, and LaTeX math; the blog pipeline supports KaTeX and syntax highlighting
- QT content is Korean devotional prose with stable section headings such as `## 말씀`, `## 관찰`, `## 적용`, and `## 기도`
- Slugs are parser-dependent: blog slugs mirror filenames, QT slugs are generated from filenames and may add numeric suffixes for duplicate dates

## ANTI-PATTERNS

- **DO NOT** switch these files to MDX — loaders expect plain markdown
- **DO NOT** rename files without considering slug impact on generated routes and sitemap output
- **DO NOT** add ad hoc frontmatter to QT files and expect the app to read it — `src/lib/qt.ts` ignores it
- **DO NOT** hand-edit AI-generated blog filenames into a different pattern — generators and loaders expect the date-prefixed scheme
- **DO NOT** move rendering concerns into markdown when the pipeline already handles math, headings, and highlighting in `src/lib/`
