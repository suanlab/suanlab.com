import PageHeader from '@/components/layout/PageHeader';
import { getAllPosts, getAllCategories, getAllTags } from '@/lib/blog';
import BlogContent from './BlogContent';

export const metadata = {
  title: 'Blog | SuanLab',
  description: '데이터 과학, 인공지능, 딥러닝에 관한 이야기를 공유하는 SuanLab 블로그입니다.',
};

export default function BlogPage() {
  const posts = getAllPosts();
  const categories = getAllCategories();
  const tags = getAllTags();

  return (
    <>
      <PageHeader
        title="Blog"
        subtitle="데이터 과학, 인공지능, 딥러닝에 관한 이야기"
        breadcrumbs={[{ label: 'Blog' }]}
      />

      <BlogContent posts={posts} categories={categories} tags={tags} />
    </>
  );
}
