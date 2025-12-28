import Link from 'next/link';
import Image from 'next/image';
import { Calendar, Folder, Rss } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { getAllPosts, getAllCategories, getAllTags } from '@/lib/blog';
import BlogFilters from './BlogFilters';

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

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="grid gap-8 lg:grid-cols-4">
            {/* 사이드바 */}
            <aside className="lg:col-span-1">
              <BlogFilters categories={categories} tags={tags} />

              {/* RSS 구독 */}
              <Card>
                <CardContent className="p-4">
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Rss className="h-4 w-4 text-primary" />
                    구독
                  </h3>
                  <a
                    href="/blog/feed.xml"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-3 py-2 rounded-md text-sm bg-orange-500 text-white hover:bg-orange-600 transition-colors"
                  >
                    <Rss className="h-4 w-4" />
                    RSS 피드 구독
                  </a>
                </CardContent>
              </Card>
            </aside>

            {/* 포스트 목록 */}
            <div className="lg:col-span-3">
              {/* 검색 결과 수 */}
              <p className="text-sm text-muted-foreground mb-6">
                {posts.length}개의 포스트
              </p>

              {/* 포스트 카드 */}
              <div className="grid gap-6 md:grid-cols-2">
                {posts.map((post) => (
                  <Link key={post.slug} href={`/blog/${post.slug}`}>
                    <Card className="h-full overflow-hidden hover:shadow-lg transition-shadow">
                      {post.thumbnail && (
                        <div className="relative aspect-video">
                          <Image
                            src={post.thumbnail}
                            alt={post.title}
                            fill
                            className="object-cover"
                          />
                        </div>
                      )}
                      <CardContent className="p-5">
                        <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
                          <Calendar className="h-3 w-3" />
                          <span>{post.date}</span>
                          <span className="mx-1">•</span>
                          <Folder className="h-3 w-3" />
                          <span>{post.category}</span>
                        </div>
                        <h2 className="font-semibold text-lg mb-2 line-clamp-2">
                          {post.title}
                        </h2>
                        <p className="text-sm text-muted-foreground line-clamp-3">
                          {post.excerpt}
                        </p>
                        <div className="flex flex-wrap gap-1 mt-3">
                          {post.tags.slice(0, 3).map((tag) => (
                            <Badge key={tag} variant="secondary" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                          {post.tags.length > 3 && (
                            <Badge variant="secondary" className="text-xs">
                              +{post.tags.length - 3}
                            </Badge>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
