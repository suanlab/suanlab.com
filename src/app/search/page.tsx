import { Metadata } from 'next';
import PageHeader from '@/components/layout/PageHeader';
import SearchClient from './SearchClient';
import { getAllPosts } from '@/lib/blog';
import { publications } from '@/data/publications';
import { lectures } from '@/data/lectures';

export const metadata: Metadata = {
  title: '검색 | SuanLab',
  description: '블로그, 논문, 강의를 검색하세요.',
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
