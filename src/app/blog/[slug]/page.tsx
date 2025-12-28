import { notFound } from 'next/navigation';
import Link from 'next/link';
import Image from 'next/image';
import { Calendar, Tag, Folder, ArrowLeft, ArrowRight, Clock } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { getPostBySlugWithHtml, getAllPosts, getPostSlugs } from '@/lib/blog';
import '@/styles/blog-prose.css';
import 'katex/dist/katex.min.css';

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const slugs = getPostSlugs();
  return slugs.map((slug) => ({
    slug: slug.replace(/\.md$/, ''),
  }));
}

export async function generateMetadata({ params }: Props) {
  const { slug } = await params;
  const post = await getPostBySlugWithHtml(slug);

  if (!post) {
    return {
      title: 'Post Not Found | SuanLab',
    };
  }

  return {
    title: `${post.title} | SuanLab Blog`,
    description: post.excerpt,
  };
}

export default async function BlogPostPage({ params }: Props) {
  const { slug } = await params;
  const post = await getPostBySlugWithHtml(slug);

  if (!post) {
    notFound();
  }

  // 이전/다음 포스트 찾기
  const allPosts = getAllPosts();
  const currentIndex = allPosts.findIndex((p) => p.slug === slug);
  const prevPost = currentIndex < allPosts.length - 1 ? allPosts[currentIndex + 1] : null;
  const nextPost = currentIndex > 0 ? allPosts[currentIndex - 1] : null;

  return (
    <>
      <PageHeader
        title={post.title}
        subtitle={post.excerpt}
        breadcrumbs={[{ label: 'Blog', href: '/blog' }, { label: post.title }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="max-w-4xl mx-auto">
            {/* 포스트 메타 정보 */}
            <Card className="mb-8">
              <CardContent className="p-6">
                <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <Calendar className="h-4 w-4" />
                    <span>{post.date}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    <span>{post.readingTime}분 소요</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Folder className="h-4 w-4" />
                    <Link href={`/blog?category=${encodeURIComponent(post.category)}`}>
                      <Badge variant="secondary">{post.category}</Badge>
                    </Link>
                  </div>
                  {post.tags.length > 0 && (
                    <div className="flex items-center gap-2">
                      <Tag className="h-4 w-4" />
                      <div className="flex flex-wrap gap-1">
                        {post.tags.map((tag) => (
                          <Link key={tag} href={`/blog?tag=${encodeURIComponent(tag)}`}>
                            <Badge variant="outline">{tag}</Badge>
                          </Link>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* 썸네일 이미지 */}
            {post.thumbnail && (
              <div className="relative aspect-video mb-8 rounded-lg overflow-hidden">
                <Image
                  src={post.thumbnail}
                  alt={post.title}
                  fill
                  className="object-cover"
                  priority
                />
              </div>
            )}

            {/* 포스트 내용 */}
            <article
              className="blog-content prose prose-lg dark:prose-invert max-w-none mb-12"
              dangerouslySetInnerHTML={{ __html: post.contentHtml }}
            />

            {/* 이전/다음 포스트 네비게이션 */}
            <div className="border-t pt-8">
              <div className="grid gap-4 md:grid-cols-2">
                {prevPost ? (
                  <Link href={`/blog/${prevPost.slug}`}>
                    <Card className="h-full hover:shadow-md transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                          <ArrowLeft className="h-4 w-4" />
                          <span>이전 글</span>
                        </div>
                        <p className="font-medium line-clamp-2">{prevPost.title}</p>
                      </CardContent>
                    </Card>
                  </Link>
                ) : (
                  <div />
                )}
                {nextPost ? (
                  <Link href={`/blog/${nextPost.slug}`}>
                    <Card className="h-full hover:shadow-md transition-shadow">
                      <CardContent className="p-4 text-right">
                        <div className="flex items-center justify-end gap-2 text-sm text-muted-foreground mb-2">
                          <span>다음 글</span>
                          <ArrowRight className="h-4 w-4" />
                        </div>
                        <p className="font-medium line-clamp-2">{nextPost.title}</p>
                      </CardContent>
                    </Card>
                  </Link>
                ) : (
                  <div />
                )}
              </div>
            </div>

            {/* 목록으로 돌아가기 */}
            <div className="text-center mt-8">
              <Link href="/blog">
                <Button variant="outline">
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  블로그 목록으로
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
