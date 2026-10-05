import { MetadataRoute } from 'next';
import { site } from '@/config/site';
import { researchAreas } from '@/data/research';
import { lectures } from '@/data/lectures';
import { playlists } from '@/data/youtube';
import { getAllPosts } from '@/lib/blog';
import { getAllQTEntries } from '@/lib/qt';
import { getAllBookPosts } from '@/lib/books';

export default function sitemap(): MetadataRoute.Sitemap {
  const staticRoutes = ['', 'suan', 'research', 'publication', 'project', 'lecture', 'youtube', 'blog', 'qt', 'book', 'book/published', 'book/online', 'course', 'deadlines', 'prompts', 'contact', 'privacy', 'terms'];
  const entries = [
    ...staticRoutes.map((route) => ({ route, date: undefined as string | undefined })),
    ...researchAreas.map((item) => ({ route: `research/${item.slug}`, date: undefined })),
    ...lectures.flatMap((item) => [
      { route: `lecture/${item.slug}`, date: undefined },
      { route: `lecture/${item.slug}/present`, date: undefined },
    ]),
    ...playlists.map((item) => ({ route: `youtube/${item.slug}`, date: undefined })),
    ...getAllPosts().map((item) => ({ route: `blog/${item.slug}`, date: item.date })),
    ...getAllQTEntries().map((item) => ({ route: `qt/${item.slug}`, date: item.date })),
    ...getAllBookPosts().map((item) => ({ route: `book/online/${item.slug}`, date: item.date })),
  ];
  return entries.map(({ route, date }) => ({
    url: `${site.url}/${route ? `${route}/` : ''}`,
    ...(date && !Number.isNaN(Date.parse(date)) ? { lastModified: new Date(date) } : {}),
    changeFrequency: route === '' ? 'weekly' : 'monthly',
    priority: route === '' ? 1 : route.includes('/') ? 0.6 : 0.8,
  }));
}
