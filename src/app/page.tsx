import { Metadata } from 'next';
import { getAllPosts } from '@/lib/blog';
import { HomepageContent } from '@/components/homepage-content';
import { mediaArticles } from '@/data/media';
import { publications } from '@/data/publications';
import { projects } from '@/data/projects';
import { lectures } from '@/data/lectures';
import { playlists } from '@/data/youtube';

const BASE_URL = 'https://suanlab.com';

export const metadata: Metadata = {
  title: { absolute: 'SuanLab | Data Science & AI Research' },
  description: '이수안 교수의 데이터 사이언스 연구실 - 딥러닝, 머신러닝, 빅데이터, 자연어처리, 컴퓨터 비전 연구 및 교육 콘텐츠 제공',
  openGraph: {
    title: 'SuanLab | Data Science & AI Research',
    description: '이수안 교수의 데이터 사이언스 연구실 - 딥러닝, 머신러닝, 빅데이터, 자연어처리, 컴퓨터 비전 연구 및 교육 콘텐츠 제공',
    url: BASE_URL,
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'SuanLab | Data Science & AI Research',
    description: '이수안 교수의 데이터 사이언스 연구실 - 딥러닝, 머신러닝, 빅데이터, 자연어처리, 컴퓨터 비전 연구 및 교육 콘텐츠 제공',
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
