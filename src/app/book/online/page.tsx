import Image from 'next/image';
import { FileText, User, Building, Calendar, ShoppingCart } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { getAllBookPostsWithHtml } from '@/lib/books';
import '@/styles/book-content.css';

export const metadata = {
  title: 'Online Book | SuanLab',
  description: 'Online book contents and materials by Professor Suan Lee',
};

export default async function OnlineBookPage() {
  const bookPosts = await getAllBookPostsWithHtml();

  return (
    <>
      <PageHeader
        title="Online Book"
        subtitle="Online book contents and materials"
        breadcrumbs={[
          { label: 'Book', href: '/book' },
          { label: 'Online Book' },
        ]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-12">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
              Book <span className="text-primary">Contents</span>
            </h2>
            <p className="mt-4 text-muted-foreground">
              Additional book materials and writings
            </p>
          </div>

          {bookPosts.length > 0 ? (
            <div className="space-y-8">
              {bookPosts.map((book) => (
                <Card key={book.slug} className="overflow-hidden">
                  <div className="flex flex-col lg:flex-row lg:items-start gap-8">
                    {/* Book Cover */}
                    {book.image && (
                      <div className="flex-shrink-0 flex justify-center lg:justify-start p-6 lg:p-8 lg:pb-0">
                        <Image
                          src={book.image}
                          alt={book.title}
                          width={200}
                          height={280}
                          className="rounded-lg shadow-lg object-contain"
                        />
                      </div>
                    )}

                    {/* Book Details */}
                    <CardContent className={`flex-1 p-6 lg:p-8 ${book.image ? 'lg:pl-0' : ''}`}>
                      <div className="mb-4">
                        <Badge variant="secondary" className="mb-3">
                          <FileText className="mr-1 h-3 w-3" />
                          Online Content
                        </Badge>
                        <h3 className="text-2xl font-bold">{book.title}</h3>
                        {book.subtitle && (
                          <p className="text-muted-foreground mt-1">{book.subtitle}</p>
                        )}
                      </div>

                      <div className="flex flex-wrap gap-4 text-sm text-muted-foreground mb-6">
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

                      {/* Book Content */}
                      <div
                        className="book-content prose prose-sm dark:prose-invert max-w-none"
                        dangerouslySetInnerHTML={{ __html: book.contentHtml }}
                      />

                      {book.url && (
                        <div className="mt-6">
                          <Button size="lg" asChild>
                            <a href={book.url} target="_blank" rel="noopener noreferrer">
                              <ShoppingCart className="mr-2 h-4 w-4" />
                              Purchase Book
                            </a>
                          </Button>
                        </div>
                      )}
                    </CardContent>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <Card className="p-12">
              <div className="flex flex-col items-center text-center">
                <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                  <FileText className="h-8 w-8 text-muted-foreground" />
                </div>
                <h3 className="text-lg font-semibold">No online books available</h3>
                <p className="mt-2 text-muted-foreground">
                  온라인 도서 콘텐츠가 준비 중입니다.
                </p>
              </div>
            </Card>
          )}
        </div>
      </section>
    </>
  );
}
