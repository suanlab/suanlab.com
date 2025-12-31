import { MetadataRoute } from 'next';
import fs from 'fs';
import path from 'path';

const BASE_URL = 'https://suanlab.com';

// Get all blog slugs
function getBlogSlugs(): string[] {
  const blogDir = path.join(process.cwd(), 'content/blog');
  try {
    const files = fs.readdirSync(blogDir);
    return files
      .filter((file) => file.endsWith('.md'))
      .map((file) => file.replace('.md', ''));
  } catch {
    return [];
  }
}

// Research area slugs
const researchSlugs = ['ds', 'dl', 'nlp', 'cv', 'graphs', 'st', 'asp'];

// Lecture slugs (from the lecture data)
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
  'PLpIPLT0Pf7IoTxTCi2MEQ94MZnHaxrP0j', // Python
  'PLpIPLT0Pf7IqSuMx237SHRdLd5ZA4AQwd', // Data Analysis
  'PLpIPLT0Pf7IqJsHj2MtpHGvXNqNbVe8ss', // ML
  'PLpIPLT0Pf7IqdpQ-gg2fBTl7ZzGx4NhnA', // DL
  'PLpIPLT0Pf7IoR5I9aKN00klvs6uVe5KXQ', // CV
  'PLpIPLT0Pf7IrcShjJiHXpWo2EhLuWilMC', // NLP
];

export default function sitemap(): MetadataRoute.Sitemap {
  const blogSlugs = getBlogSlugs();

  // Static pages
  const staticPages: MetadataRoute.Sitemap = [
    {
      url: BASE_URL,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 1,
    },
    {
      url: `${BASE_URL}/suan`,
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.9,
    },
    {
      url: `${BASE_URL}/research`,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 0.8,
    },
    {
      url: `${BASE_URL}/publication`,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 0.8,
    },
    {
      url: `${BASE_URL}/project`,
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.7,
    },
    {
      url: `${BASE_URL}/lecture`,
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.8,
    },
    {
      url: `${BASE_URL}/youtube`,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 0.8,
    },
    {
      url: `${BASE_URL}/blog`,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 0.8,
    },
    {
      url: `${BASE_URL}/book`,
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.7,
    },
    {
      url: `${BASE_URL}/book/published`,
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.6,
    },
    {
      url: `${BASE_URL}/book/online`,
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.6,
    },
    {
      url: `${BASE_URL}/course`,
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.7,
    },
  ];

  // Research area pages
  const researchPages: MetadataRoute.Sitemap = researchSlugs.map((slug) => ({
    url: `${BASE_URL}/research/${slug}`,
    lastModified: new Date(),
    changeFrequency: 'monthly' as const,
    priority: 0.7,
  }));

  // Lecture pages
  const lecturePages: MetadataRoute.Sitemap = lectureSlugs.map((slug) => ({
    url: `${BASE_URL}/lecture/${slug}`,
    lastModified: new Date(),
    changeFrequency: 'monthly' as const,
    priority: 0.6,
  }));

  // Blog pages
  const blogPages: MetadataRoute.Sitemap = blogSlugs.map((slug) => ({
    url: `${BASE_URL}/blog/${slug}`,
    lastModified: new Date(),
    changeFrequency: 'weekly' as const,
    priority: 0.6,
  }));

  // YouTube playlist pages
  const youtubePages: MetadataRoute.Sitemap = youtubePlaylistIds.map((id) => ({
    url: `${BASE_URL}/youtube/${id}`,
    lastModified: new Date(),
    changeFrequency: 'weekly' as const,
    priority: 0.5,
  }));

  return [
    ...staticPages,
    ...researchPages,
    ...lecturePages,
    ...blogPages,
    ...youtubePages,
  ];
}
