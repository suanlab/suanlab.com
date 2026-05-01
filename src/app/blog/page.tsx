import { Metadata } from 'next';
import PageHeader from '@/components/layout/PageHeader';
import { getAllPosts, getAllCategories, getAllTags } from '@/lib/blog';
import BlogContent from './BlogContent';

const BASE_URL = 'https://suanlab.com';

export const metadata: Metadata = {
  title: 'Blog',
  description: '데이터 과학, 인공지능, 딥러닝에 관한 이야기를 공유하는 SuanLab 블로그입니다.',
  openGraph: {
    title: 'Blog | SuanLab',
    description: '데이터 과학, 인공지능, 딥러닝에 관한 이야기를 공유하는 SuanLab 블로그입니다.',
    url: `${BASE_URL}/blog`,
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Blog | SuanLab',
    description: '데이터 과학, 인공지능, 딥러닝에 관한 이야기를 공유하는 SuanLab 블로그입니다.',
  },
  alternates: {
    canonical: `${BASE_URL}/blog/`,
  },
};

export default function BlogPage() {
  const posts = getAllPosts();
  const categories = getAllCategories();
  const tags = getAllTags();

  return (
    <>
      <PageHeader
        title="Blog"
        subtitleKey="pageheader.blog.subtitle"
        breadcrumbs={[{ label: 'Blog' }]}
      />

      <BlogContent posts={posts} categories={categories} tags={tags} />
    </>
  );
}
