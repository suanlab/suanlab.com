import { Metadata } from 'next';
import PageHeader from '@/components/layout/PageHeader';
import SearchClient from './SearchClient';
import { getAllPosts } from '@/lib/blog';
import { publications } from '@/data/publications';
import { lectures } from '@/data/lectures';

const BASE_URL = 'https://suanlab.com';

export const metadata: Metadata = {
  title: '검색',
  description: 'SuanLab에서 블로그, 논문, 강의를 검색하세요.',
  openGraph: {
    title: '검색 | SuanLab',
    description: 'SuanLab에서 블로그, 논문, 강의를 검색하세요.',
    url: `${BASE_URL}/search`,
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: '검색 | SuanLab',
    description: 'SuanLab에서 블로그, 논문, 강의를 검색하세요.',
  },
  alternates: {
    canonical: `${BASE_URL}/search/`,
  },
  robots: {
    index: false,
    follow: true,
  },
};

export default function SearchPage() {
  // Load all searchable data at build time
  const posts = getAllPosts();

  return (
    <>
      <PageHeader
        title="검색"
        subtitle="블로그, 논문, 강의를 검색하세요"
        breadcrumbs={[{ label: '검색' }]}
      />
      <SearchClient posts={posts} publications={publications} lectures={lectures} />
    </>
  );
}
