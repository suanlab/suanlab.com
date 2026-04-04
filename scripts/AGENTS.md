# SCRIPTS — CLI Tools, Bots, Extraction

## OVERVIEW

Standalone CLI tools run outside Next.js. Blog generation via AI, Slack bot automation, legacy-data extraction, and asset generation. Uses `tsx` for TypeScript execution.

## STRUCTURE

```
scripts/
├── blog/
│   ├── generate.ts          # Main CLI entry: `npm run blog:generate` (Commander.js)
│   ├── topic-generator.ts   # Topic → markdown blog post via AI
│   ├── paper-summarizer.ts  # arXiv/PDF → paper review post via AI
│   └── tsconfig.json        # Separate tsconfig (target ES2020, output dist/scripts)
├── slack-bot.ts             # Slack slash-command bot; can generate and push posts
├── extract-courses.js       # Extract course data from legacy WWW site
├── extract-projects.js      # Extract project data from legacy WWW site
├── extract-publications.js  # Extract publication data from legacy WWW site
├── extract-youtube.js       # Extract YouTube data from legacy WWW site
├── generate-book-covers.ts  # Generate book cover images (Sharp)
├── generate-favicon.ts      # Generate favicon variants
├── generate-og-image.ts     # Generate OpenGraph images
└── convert-favicon.ts       # Convert favicon formats
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add blog generation mode | `blog/generate.ts` | Commander.js subcommands |
| Modify AI prompts | `../src/lib/ai/prompts.ts` | Templates used by generators |
| Paper processing | `blog/paper-summarizer.ts` | arXiv → PDF → chunks → AI summary |
| Slack automation | `slack-bot.ts` | Bolt app, batch arXiv mode, git push flow |
| Extract legacy data | `extract-*.js` | One-time migration scripts (JS, not TS) |

## CONVENTIONS

- **Run with tsx**: `npx tsx scripts/{file}.ts` or via npm scripts
- **Blog scripts import from `../src/lib/`** — reuse AI and PDF libraries
- **Slack bot loads `.env.local` first** — credentials are read before Bolt app startup
- **Slack bot shells out to git** — it can commit/push generated posts from the repo root
- **Extract scripts are JS** (not TS) — legacy one-time migration tools
- **Environment**: Scripts read `.env.local` via dotenv
- **Blog output**: Saves to `content/blog/YYYYMMDD-slug.md` with frontmatter
- **Interactive CLI**: Uses Commander.js plus Node `readline` prompts

## ANTI-PATTERNS

- **DO NOT** import script entry points into `src/` runtime code — keep scripts as standalone tooling
- **DO NOT** run extract-*.js again — they were one-time migrations from WWW
- **DO NOT** run bots without `.env.local` configured — will crash on missing tokens
- **DO NOT** assume a Telegram bot exists — only `slack-bot.ts` is present on disk
- **DO NOT** run `slack-bot.ts` outside the repo root — it uses `process.cwd()` for `.env.local` and git commands
