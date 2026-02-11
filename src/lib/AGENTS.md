# LIB — Business Logic Layer

## OVERVIEW

Core libraries for content processing (markdown → HTML), AI text/image generation, and PDF/arXiv integration. All content types share the unified.js pipeline pattern.

## STRUCTURE

```
lib/
├── blog.ts              # Blog post CRUD + markdown pipeline (most complex)
├── qt.ts                # QT devotional CRUD + Bible reference parsing
├── books.ts             # Book content CRUD
├── utils.ts             # Single fn: cn() for Tailwind class merging
├── ai/
│   ├── claude.ts        # AI orchestration: OpenAI GPT-4o + Gemini (fallback)
│   ├── prompts.ts       # Prompt templates: topic, paper summary, synthesis
│   ├── image.ts         # Gemini image generation for thumbnails
│   └── types.ts         # TopicGeneratorOptions, PaperSummarizerOptions, GeneratedPost
└── pdf/
    ├── parser.ts        # PDF text extraction + chunking (~100k chars/chunk)
    ├── arxiv.ts         # arXiv metadata fetch + PDF download (retry + rate limit)
    └── figure-extractor.ts  # Extract figures from PDFs
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add markdown feature | `blog.ts` | Modify unified pipeline (remark → rehype chain) |
| New content type | Copy `books.ts` | Simplest pattern: gray-matter + unified |
| Change AI model | `ai/claude.ts` | `generateWithOpenAI`, `generateWithGemini` |
| Modify prompts | `ai/prompts.ts` | `TOPIC_PROMPT_TEMPLATE`, `PAPER_SUMMARY_PROMPT` |
| PDF text extraction | `pdf/parser.ts` | `parsePdfFromUrl`, `parsePdfFromBuffer` |
| arXiv integration | `pdf/arxiv.ts` | `fetchArxivMetadata` with 3-retry, 429 handling |

## CONVENTIONS

- **Content loaders follow same pattern**: `getAll*()`, `get*BySlug()`, `get*BySlugWithHtml()`
- **Markdown pipeline**: remarkParse → remarkGfm → remarkMath → remarkRehype → rehypeSlug → rehypeAutolinkHeadings → rehypeKatex → rehypeHighlight → rehypeStringify
- **AI fallback**: OpenAI first → Gemini on failure (in `generateWithDualAI`)
- **OpenAI client**: Singleton pattern via `getOpenAIClient()`
- **Refusal detection**: `isRefusalMessage()` checks 8 known refusal patterns
- **Error handling**: throw + try-catch (no centralized logger)
- **arXiv retries**: 3 attempts, 10s backoff, 30s on 429

## ANTI-PATTERNS

- **DO NOT** add database queries — all content reads from filesystem
- **DO NOT** use MDX plugins — content is plain markdown only
- **DO NOT** remove math/katex plugins — blog posts contain LaTeX equations
- **DO NOT** import from `ai/` in client components — these use Node.js APIs (process.env, fs)
