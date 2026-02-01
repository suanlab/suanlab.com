import { notFound } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, ArrowRight, Book, BookOpen, Calendar, Home } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { getQTBySlug, getQTSlugs, getAllQTEntries, BIBLE_BOOKS, type BibleBook } from '@/lib/qt';
import type { Metadata } from 'next';

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const slugs = getQTSlugs();
  return slugs.map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const entry = getQTBySlug(slug);

  if (!entry) {
    return { title: 'QT Not Found | SuanLab' };
  }

  return {
    title: `${entry.title} | Quiet Time`,
    description: entry.reflection.slice(0, 160) + '...',
    openGraph: {
      title: entry.title,
      description: entry.reflection.slice(0, 160) + '...',
      type: 'article',
    },
  };
}

export default async function QTEntryPage({ params }: Props) {
  const { slug } = await params;
  const entry = getQTBySlug(slug);

  if (!entry) {
    notFound();
  }

  // Get prev/next entries
  const allEntries = getAllQTEntries();
  const currentIndex = allEntries.findIndex(e => e.slug === slug);
  const prevEntry = currentIndex < allEntries.length - 1 ? allEntries[currentIndex + 1] : null;
  const nextEntry = currentIndex > 0 ? allEntries[currentIndex - 1] : null;

  const bookInfo = entry.bibleBook ? BIBLE_BOOKS[entry.bibleBook as BibleBook] : null;

  return (
    <div className="min-h-screen bg-gradient-to-b from-amber-50/50 to-background dark:from-amber-950/20">
      {/* Header */}
      <header className="border-b bg-background/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="container py-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Link href="/qt" className="hover:text-primary flex items-center gap-1">
              <Home className="h-4 w-4" />
              <span>QT</span>
            </Link>
            {entry.bibleBook && (
              <>
                <span>/</span>
                <span>{entry.bibleBook}</span>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="container py-8 max-w-3xl mx-auto">
        {/* Entry Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            {bookInfo && (
              <Badge
                variant="outline"
                className={bookInfo.testament === 'old'
                  ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                  : 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
                }
              >
                {bookInfo.testament === 'old' ? '구약' : '신약'}
              </Badge>
            )}
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Calendar className="h-4 w-4" />
              <span>{entry.date}</span>
            </div>
          </div>

          <h1 className="text-2xl md:text-3xl font-bold mb-4">{entry.title}</h1>

          {entry.bibleReference && (
            <div className="flex items-center gap-2 text-amber-700 dark:text-amber-400">
              <Book className="h-5 w-5" />
              <span className="font-medium">{entry.bibleReference}</span>
            </div>
          )}
        </div>

        {/* Bible Verses */}
        {entry.content && (
          <Card className="mb-8 bg-amber-50/50 dark:bg-amber-950/30 border-amber-200 dark:border-amber-800">
            <CardContent className="p-6">
              <div className="flex items-center gap-2 mb-4 text-amber-700 dark:text-amber-400">
                <BookOpen className="h-5 w-5" />
                <span className="font-semibold">본문 말씀</span>
              </div>
              <div className="space-y-2 text-foreground/90 leading-relaxed whitespace-pre-line font-serif">
                {entry.content}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Reflection */}
        <Card className="mb-12">
          <CardContent className="p-6">
            <div className="flex items-center gap-2 mb-4 text-primary">
              <span className="text-lg">✍️</span>
              <span className="font-semibold">묵상</span>
            </div>
            <div className="prose prose-lg dark:prose-invert max-w-none leading-relaxed whitespace-pre-line">
              {entry.reflection}
            </div>
          </CardContent>
        </Card>

        {/* Navigation */}
        <div className="border-t pt-8">
          <div className="grid gap-4 md:grid-cols-2">
            {prevEntry ? (
              <Link href={`/qt/${prevEntry.slug}`}>
                <Card className="h-full hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                      <ArrowLeft className="h-4 w-4" />
                      <span>이전 묵상</span>
                    </div>
                    <p className="font-medium line-clamp-2">{prevEntry.title}</p>
                    <p className="text-xs text-muted-foreground mt-1">{prevEntry.date}</p>
                  </CardContent>
                </Card>
              </Link>
            ) : (
              <div />
            )}
            {nextEntry ? (
              <Link href={`/qt/${nextEntry.slug}`}>
                <Card className="h-full hover:shadow-md transition-shadow">
                  <CardContent className="p-4 text-right">
                    <div className="flex items-center justify-end gap-2 text-sm text-muted-foreground mb-2">
                      <span>다음 묵상</span>
                      <ArrowRight className="h-4 w-4" />
                    </div>
                    <p className="font-medium line-clamp-2">{nextEntry.title}</p>
                    <p className="text-xs text-muted-foreground mt-1">{nextEntry.date}</p>
                  </CardContent>
                </Card>
              </Link>
            ) : (
              <div />
            )}
          </div>
        </div>

        {/* Back to list */}
        <div className="text-center mt-8">
          <Link href="/qt">
            <Button variant="outline">
              <ArrowLeft className="h-4 w-4 mr-2" />
              전체 목록으로
            </Button>
          </Link>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t py-8 mt-12">
        <div className="container text-center text-sm text-muted-foreground">
          <p>Suan Lee</p>
          <p className="mt-1">매일 말씀 묵상과 함께하는 신앙 여정</p>
        </div>
      </footer>
    </div>
  );
}
