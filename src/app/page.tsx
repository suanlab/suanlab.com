import { featuredPublications, featuredProjectIds } from '@/data/editorial';
import { site } from '@/config/site';
import { Metadata } from 'next';
import { getAllPosts } from '@/lib/blog';
import { HomepageContent } from '@/components/homepage-content';
import { mediaArticles } from '@/data/media';
import { publications } from '@/data/publications';
import { projects } from '@/data/projects';
import { lectures } from '@/data/lectures';
import { playlists } from '@/data/youtube';

const BASE_URL = site.url;

export const metadata: Metadata = {
  title: { absolute: site.title },
  description: site.description,
  openGraph: {
    title: site.title,
    description: site.description,
    url: BASE_URL,
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: site.title,
    description: site.description,
  },
  alternates: {
    canonical: `${BASE_URL}/`,
  },
};

export default async function Home() {
  const recentPosts = getAllPosts().slice(0, 6);

  return (
    <HomepageContent
      recentPosts={recentPosts}
      selectedPublications={featuredPublications.map(selection => ({ ...publications.find(publication => publication.id === selection.id)!, ...selection }))}
      featuredProjects={featuredProjectIds.map(id => projects.find(project => project.id === id)!)}
      stats={{
        publications: publications.length,
        videos: playlists.reduce((acc, playlist) => acc + playlist.videos.length, 0),
        projects: projects.length,
        lectures: lectures.length,
      }}
      mediaArticles={mediaArticles}
    />
  );
}
