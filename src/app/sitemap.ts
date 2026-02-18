import { MetadataRoute } from 'next';
import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

const BASE_URL = 'https://suanlab.com';

// Get blog slugs with their dates from frontmatter
function getBlogEntries(): { slug: string; date: string }[] {
  const blogDir = path.join(process.cwd(), 'content/blog');
  try {
    const files = fs.readdirSync(blogDir).filter((f) => f.endsWith('.md'));
    return files.map((file) => {
      const slug = file.replace('.md', '');
      try {
        const raw = fs.readFileSync(path.join(blogDir, file), 'utf-8');
        const { data } = matter(raw);
        return { slug, date: data.date || slug.slice(0, 8) };
      } catch {
        return { slug, date: slug.slice(0, 8) };
      }
    });
  } catch {
    return [];
  }
}

// Get QT slugs with their dates from filename (YYYY-MM-DD Title.md)
function getQTEntries(): { slug: string; date: string }[] {
  const qtDir = path.join(process.cwd(), 'content/qt');
  try {
    const files = fs.readdirSync(qtDir).filter((f) => f.endsWith('.md'));
    return files.map((file) => {
      const slug = file.replace('.md', '');
      const dateMatch = slug.match(/^(\d{4}-\d{2}-\d{2})/);
      const date = dateMatch ? dateMatch[1] : new Date().toISOString().split('T')[0];
      return { slug: encodeURIComponent(slug), date };
    });
  } catch {
    return [];
  }
}

// Research area slugs
const researchSlugs = ['ds', 'dl', 'nlp', 'cv', 'graphs', 'st', 'asp'];

// Lecture slugs
const lectureSlugs = [
  'python-for-data-analysis',
  'ml-perfect-guide',
  'tensorflow-keras-deeplearning',
  'practical-datascience',
  'ai-programming',
  'python-data-visualization',
  'bigdata-analysis',
  'python-web-crawling',
  'deep-learning-intro',
  'data-analysis-basic',
  'python-basic',
  'text-mining',
  'pytorch-tutorial',
  'gnn-tutorial',
  'transformers-tutorial',
  'langchain-tutorial',
];

// YouTube playlist IDs
const youtubePlaylistIds = [
  'PLpIPLT0Pf7IoTxTCi2MEQ94MZnHaxrP0j',
  'PLpIPLT0Pf7IqSuMx237SHRdLd5ZA4AQwd',
  'PLpIPLT0Pf7IqJsHj2MtpHGvXNqNbVe8ss',
  'PLpIPLT0Pf7IqdpQ-gg2fBTl7ZzGx4NhnA',
  'PLpIPLT0Pf7IoR5I9aKN00klvs6uVe5KXQ',
  'PLpIPLT0Pf7IrcShjJiHXpWo2EhLuWilMC',
];

export default function sitemap(): MetadataRoute.Sitemap {
  const blogEntries = getBlogEntries();
  const qtEntries = getQTEntries();

  // Static pages (trailing slash for GitHub Pages compatibility)
  const staticPages: MetadataRoute.Sitemap = [
    {
      url: `${BASE_URL}/`,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 1,
    },
    {
      url: `${BASE_URL}/suan/`,
      lastModified: '2025-01-01',
      changeFrequency: 'monthly',
      priority: 0.9,
    },
    {
      url: `${BASE_URL}/research/`,
      lastModified: '2025-01-01',
      changeFrequency: 'weekly',
      priority: 0.8,
    },
    {
      url: `${BASE_URL}/publication/`,
      lastModified: '2025-01-01',
      changeFrequency: 'weekly',
      priority: 0.8,
    },
    {
      url: `${BASE_URL}/project/`,
      lastModified: '2025-01-01',
      changeFrequency: 'monthly',
      priority: 0.7,
    },
    {
      url: `${BASE_URL}/lecture/`,
      lastModified: '2025-01-01',
      changeFrequency: 'monthly',
      priority: 0.8,
    },
    {
      url: `${BASE_URL}/youtube/`,
      lastModified: '2025-06-01',
      changeFrequency: 'weekly',
      priority: 0.8,
    },
    {
      url: `${BASE_URL}/blog/`,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 0.8,
    },
    {
      url: `${BASE_URL}/qt/`,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 0.7,
    },
    {
      url: `${BASE_URL}/book/`,
      lastModified: '2025-01-01',
      changeFrequency: 'monthly',
      priority: 0.7,
    },
    {
      url: `${BASE_URL}/book/published/`,
      lastModified: '2025-01-01',
      changeFrequency: 'monthly',
      priority: 0.6,
    },
    {
      url: `${BASE_URL}/book/online/`,
      lastModified: '2025-01-01',
      changeFrequency: 'monthly',
      priority: 0.6,
    },
    {
      url: `${BASE_URL}/course/`,
      lastModified: '2025-03-01',
      changeFrequency: 'monthly',
      priority: 0.7,
    },
  ];

  // Research area pages
  const researchPages: MetadataRoute.Sitemap = researchSlugs.map((slug) => ({
    url: `${BASE_URL}/research/${slug}/`,
    lastModified: '2025-01-01',
    changeFrequency: 'monthly' as const,
    priority: 0.7,
  }));

  // Lecture pages
  const lecturePages: MetadataRoute.Sitemap = lectureSlugs.map((slug) => ({
    url: `${BASE_URL}/lecture/${slug}/`,
    lastModified: '2025-01-01',
    changeFrequency: 'monthly' as const,
    priority: 0.6,
  }));

  // Blog pages — use actual post date as lastModified
  const blogPages: MetadataRoute.Sitemap = blogEntries.map(({ slug, date }) => ({
    url: `${BASE_URL}/blog/${slug}/`,
    lastModified: date,
    changeFrequency: 'monthly' as const,
    priority: 0.6,
  }));

  // QT pages — use actual date from filename
  const qtPages: MetadataRoute.Sitemap = qtEntries.map(({ slug, date }) => ({
    url: `${BASE_URL}/qt/${slug}/`,
    lastModified: date,
    changeFrequency: 'yearly' as const,
    priority: 0.4,
  }));

  // YouTube playlist pages
  const youtubePages: MetadataRoute.Sitemap = youtubePlaylistIds.map((id) => ({
    url: `${BASE_URL}/youtube/${id}/`,
    lastModified: '2025-06-01',
    changeFrequency: 'weekly' as const,
    priority: 0.5,
  }));

  return [
    ...staticPages,
    ...researchPages,
    ...lecturePages,
    ...blogPages,
    ...qtPages,
    ...youtubePages,
  ];
}
