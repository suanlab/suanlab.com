import Image from 'next/image';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { ArrowLeft, User, Building, Calendar, ShoppingCart, BookOpen } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { getBookBySlugWithHtml, getBookSlugs } from '@/lib/books';
import '@/styles/book-content.css';

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const slugs = getBookSlugs();
  return slugs.map((slug) => ({
    slug: slug.replace(/\.md$/, ''),
  }));
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const book = await getBookBySlugWithHtml(slug);

  if (!book) {
    return { title: 'Book Not Found | SuanLab' };
  }

  return {
    title: `${book.title} | SuanLab`,
    description: book.subtitle || `${book.title} by ${book.author}`,
  };
}

export default async function BookDetailPage({ params }: PageProps) {
  const { slug } = await params;
  const book = await getBookBySlugWithHtml(slug);

  if (!book) {
    notFound();
  }

  return (
    <>
      <PageHeader
        title={book.title}
        subtitle={book.subtitle}
        breadcrumbs={[
          { label: 'Book', href: '/book' },
          { label: 'Online Book', href: '/book/online' },
          { label: book.title },
        ]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="mb-8">
            <Button variant="ghost" asChild>
              <Link href="/book/online">
                <ArrowLeft className="mr-2 h-4 w-4" />
                책장으로 돌아가기
              </Link>
            </Button>
          </div>

          <Card className="overflow-hidden">
            <div className="flex flex-col lg:flex-row lg:items-start gap-8">
              {/* Book Cover */}
              <div className="flex-shrink-0 flex justify-center lg:justify-start p-6 lg:p-8 lg:pb-0">
                {book.image ? (
                  <Image
                    src={book.image}
                    alt={book.title}
                    width={250}
                    height={350}
                    className="rounded-lg shadow-2xl object-contain"
                  />
                ) : (
                  <div className="w-[250px] h-[350px] bg-gradient-to-br from-primary/20 to-primary/5 rounded-lg shadow-2xl flex items-center justify-center">
                    <BookOpen className="w-16 h-16 text-primary/50" />
                  </div>
                )}
              </div>

              {/* Book Details */}
              <CardContent className={`flex-1 p-6 lg:p-8 ${book.image ? 'lg:pl-0' : ''}`}>
                <div className="mb-6">
                  <Badge variant="secondary" className="mb-3">
                    <BookOpen className="mr-1 h-3 w-3" />
                    Online Book
                  </Badge>
                  <h1 className="text-3xl font-bold">{book.title}</h1>
                  {book.subtitle && (
                    <p className="text-xl text-muted-foreground mt-2">{book.subtitle}</p>
                  )}
                </div>

                <div className="flex flex-wrap gap-4 text-sm text-muted-foreground mb-8">
                  <div className="flex items-center gap-1">
                    <User className="h-4 w-4" />
                    <span>{book.author}</span>
                  </div>
                  {book.publisher && (
                    <div className="flex items-center gap-1">
                      <Building className="h-4 w-4" />
                      <span>{book.publisher}</span>
                    </div>
                  )}
                  <div className="flex items-center gap-1">
                    <Calendar className="h-4 w-4" />
                    <span>{book.date}</span>
                  </div>
                </div>

                {book.url && (
                  <div className="mb-8">
                    <Button size="lg" asChild>
                      <a href={book.url} target="_blank" rel="noopener noreferrer">
                        <ShoppingCart className="mr-2 h-4 w-4" />
                        구매하기
                      </a>
                    </Button>
                  </div>
                )}
              </CardContent>
            </div>
          </Card>

          {/* Book Content */}
          <Card className="mt-8">
            <CardContent className="p-6 lg:p-10">
              <div
                className="book-content prose prose-lg dark:prose-invert max-w-none"
                dangerouslySetInnerHTML={{ __html: book.contentHtml }}
              />
            </CardContent>
          </Card>
        </div>
      </section>
    </>
  );
}
