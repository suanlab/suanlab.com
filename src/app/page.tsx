import { getAllPosts } from '@/lib/blog';
import { HomepageContent } from '@/components/homepage-content';
import { mediaArticles } from '@/data/media';
import { publications } from '@/data/publications';
import { projects } from '@/data/projects';
import { lectures } from '@/data/lectures';
import { playlists } from '@/data/youtube';

export default async function Home() {
  const recentPosts = getAllPosts().slice(0, 6);

  return (
    <HomepageContent
      recentPosts={recentPosts}
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
