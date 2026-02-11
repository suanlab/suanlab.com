# SCRIPTS — CLI Tools, Bots, Extraction

## OVERVIEW

Standalone CLI tools run outside Next.js. Blog generation via AI, bot integrations (Telegram/Slack), data extraction from legacy site, and asset generation. Uses `tsx` for TypeScript execution.

## STRUCTURE

```
scripts/
├── blog/
│   ├── generate.ts          # Main CLI entry: `npm run blog:generate` (Commander.js)
│   ├── topic-generator.ts   # Topic → markdown blog post via AI
│   ├── paper-summarizer.ts  # arXiv/PDF → paper review post via AI
│   └── tsconfig.json        # Separate tsconfig (target ES2020, output dist/scripts)
├── telegram-bot.ts          # Telegram bot (polling mode, user whitelist)
├── slack-bot.ts             # Slack bot integration
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
| Telegram commands | `telegram-bot.ts` | Polling mode, TELEGRAM_ALLOWED_USERS whitelist |
| Extract legacy data | `extract-*.js` | One-time migration scripts (JS, not TS) |

## CONVENTIONS

- **Run with tsx**: `npx tsx scripts/{file}.ts` or via npm scripts
- **Blog scripts import from `../src/lib/`** — reuse AI and PDF libraries
- **Extract scripts are JS** (not TS) — legacy one-time migration tools
- **Environment**: Scripts read `.env.local` via dotenv
- **Blog output**: Saves to `content/blog/YYYYMMDD-slug.md` with frontmatter
- **Interactive CLI**: Uses Commander.js + Inquirer + Ora (spinner)

## ANTI-PATTERNS

- **DO NOT** import scripts from `src/` — scripts are standalone entry points
- **DO NOT** run extract-*.js again — they were one-time migrations from WWW
- **DO NOT** run bots without `.env.local` configured — will crash on missing tokens
