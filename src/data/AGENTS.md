# DATA — Static Content & Configuration

## OVERVIEW

TypeScript data files exporting arrays/objects consumed by pages at build time. No database — all content is code. Largest files are publications (2672 lines) and youtube (1311 lines).

## STRUCTURE

```
data/
├── navigation.ts              # Nav structure, social links, contact info, site description
├── networks.ts                # Professional network categories (459 lines)
├── awards.ts                  # Awards and honors
├── media.ts                   # Media coverage articles
├── news.ts                    # News items
├── overseas-experiences.ts    # International experiences
├── publications/index.ts      # 271 publications across 8 types (2672 lines)
├── courses/index.ts           # Course + seminar data (1062 lines)
├── lectures/index.ts          # Lecture data (350 lines)
├── projects/index.ts          # Project portfolio (708 lines)
├── research/index.ts          # 7 research areas with topics, tech, achievements (509 lines)
├── youtube/index.ts           # YouTube playlists and videos (1311 lines)
├── academic-activities/index.ts  # Academic committees, reviews (751 lines)
└── blog/                      # Blog metadata helpers
```

## CONVENTIONS

- **Each file exports typed arrays/objects** — consumed directly by page components
- **Interfaces defined inline** in each file (not in types/)
- **Publication types**: journal, conference, book, patent, thesis, domestic-journal, domestic-conference, report
- **Research areas**: 7 slugs — data-science, deep-learning, nlp, computer-vision, graphs, spatio-temporal, audio
- **Navigation**: `NavItem[]` with nested `children` — drives header menu
- **Dates**: String format `"YYYY"` or `"YYYY-MM"` or `"YYYY.MM.DD"`

## ANTI-PATTERNS

- **DO NOT** import these in `lib/` — data flows one way: data → pages
- **DO NOT** add async data fetching — all data is static, synchronous exports
- **DO NOT** split large files into per-item files — the array-export pattern is intentional
