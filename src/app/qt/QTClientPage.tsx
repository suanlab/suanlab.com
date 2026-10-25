'use client';
import LocalizedContent, { OriginalLanguageNotice } from '@/components/localized-content';
import { englishText } from '@/data/i18n/translate';
import SourceText from '@/components/source-text';

import { MotionList, MotionItem } from '@/components/motion/motion-list';
import { updateListQuery } from '@/lib/list-query';
import MenuDotField from '@/components/layout/PageHeader/MenuDotField';
import { useLanguage } from '@/components/language-provider';
import { useState, useMemo, useEffect } from 'react';
import Link from 'next/link';
import { Book, BookOpen, Calendar, ChevronRight, Search } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface QTEntry {
  slug: string;
  date: string;
  title: string;
  bibleBook?: string;
  chapter?: number;
  bibleReference?: string;
}

interface QTByBook {
  book: string;
  bookInfo: { order: number; abbr: string; testament: string };
  entries: QTEntry[];
}

interface QTStats {
  total: number;
  booksCount: number;
  oldTestamentBooks: number;
  newTestamentBooks: number;
  oldTestamentEntries: number;
  newTestamentEntries: number;
  years: string[];
  dateRange: { start: string; end: string };
}

interface Props {
  byBook: QTByBook[];
  stats: QTStats;
  recentEntries: QTEntry[];
}

export default function QTClientPage({ byBook, stats, recentEntries }: Props) {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [filters, setFilters] = useState({ q: '', book: '', testament: 'all', tab: 'bybook' });
  const { q: searchQuery, book: selectedBook, testament: selectedTestament } = filters;
  useEffect(() => {
    const sync = () => {
      const params = new URLSearchParams(window.location.search);
      const testament = params.get('testament');
      setFilters({ q: params.get('q') ?? '', book: params.get('book') ?? '',
        testament: testament === 'old' || testament === 'new' ? testament : 'all',
        tab: params.get('tab') === 'recent' ? 'recent' : 'bybook' });
    };
    sync(); window.addEventListener('popstate', sync);
    return () => window.removeEventListener('popstate', sync);
  }, []);
  function updateFilters(patch: Partial<typeof filters>, push = true) {
    const next = { ...filters, ...patch }; setFilters(next);
    const url = updateListQuery(new URL(window.location.href), {
      q: next.q, book: next.book, testament: next.testament === 'all' ? null : next.testament,
      tab: next.tab === 'bybook' ? null : next.tab,
    });
    window.history[push ? 'pushState' : 'replaceState'](null, '', url);
  }
  const query = searchQuery.trim().toLowerCase();
  const matchesEntry = (entry: QTEntry, book: string) => !query || `${book} ${englishText(book)} ${entry.title} ${entry.bibleReference ?? ''} ${entry.date}`.toLowerCase().includes(query);
  const filteredBooks = useMemo(() => byBook
    .filter(book => (!selectedBook || book.book === selectedBook) && (selectedTestament === 'all' || book.bookInfo.testament === selectedTestament))
    .map(book => ({ ...book, entries: book.entries.filter(entry => !query || `${book.book} ${englishText(book.book)} ${entry.title} ${entry.bibleReference ?? ''} ${entry.date}`.toLowerCase().includes(query)) }))
    .filter(book => book.entries.length > 0), [byBook, selectedBook, selectedTestament, query]);
  const filteredRecent = recentEntries.filter(entry => (!selectedBook || entry.bibleBook === selectedBook)
    && (selectedTestament === 'all' || byBook.find(book => book.book === entry.bibleBook)?.bookInfo.testament === selectedTestament)
    && matchesEntry(entry, entry.bibleBook ?? ''));
  const resultCount = filters.tab === 'recent' ? filteredRecent.length : filteredBooks.reduce((total, book) => total + book.entries.length, 0);

  return <LocalizedContent>
    <div className="min-h-screen bg-gradient-to-b from-amber-50/50 to-background dark:from-amber-950/20">
      {/* Header */}
      <header className="qt-menu-header overflow-hidden border-b bg-background/80 backdrop-blur-sm sticky top-0 z-10">
        <MenuDotField variant="qt" />
        <div className="container relative z-10 py-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-amber-100 dark:bg-amber-900 rounded-lg">
              <BookOpen className="h-8 w-8 text-amber-700 dark:text-amber-300" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold">Quiet Time</h1>
              <p className="text-muted-foreground text-sm">{ko ? '성경과 함께하는 묵상 일지' : 'Reflections on the Bible'}</p>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-2xl font-bold text-amber-600 dark:text-amber-400">{stats.total}</p>
                <p className="text-xs text-muted-foreground">{ko ? '총 묵상' : 'Reflections'}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{stats.booksCount}</p>
                <p className="text-xs text-muted-foreground">{ko ? '성경 책' : 'Bible books'}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">{stats.oldTestamentEntries}</p>
                <p className="text-xs text-muted-foreground">{ko ? '구약' : 'Old Testament'}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">{stats.newTestamentEntries}</p>
                <p className="text-xs text-muted-foreground">{ko ? '신약' : 'New Testament'}</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </header>

      <OriginalLanguageNotice />
      <section aria-labelledby="qt-list-title" className="container py-8">
        <h2 id="qt-list-title" className="sr-only">{ko ? '묵상 목록' : 'Reflections'}</h2>
        {/* Search and Filter */}
        <div className="flex flex-col md:flex-row gap-4 mb-8">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder={ko ? '묵상 제목 또는 성경 책 검색...' : 'Search reflections or Bible books...'} aria-label={ko ? '묵상 검색' : 'Search reflections'}
              className="pl-10"
              value={searchQuery}
              onChange={(e) => updateFilters({ q: e.target.value, tab: 'bybook' }, false)}
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              variant={selectedTestament === 'all' ? 'default' : 'outline'}
              size="sm"
              onClick={() => updateFilters({ testament: 'all' })} aria-pressed={selectedTestament === 'all'}
            >
              {ko ? '전체' : 'All'}
            </Button>
            <Button
              variant={selectedTestament === 'old' ? 'default' : 'outline'}
              size="sm"
              onClick={() => updateFilters({ testament: 'old' })} aria-pressed={selectedTestament === 'old'}
            >
              {ko ? '구약' : 'Old Testament'}
            </Button>
            <Button
              variant={selectedTestament === 'new' ? 'default' : 'outline'}
              size="sm"
              onClick={() => updateFilters({ testament: 'new' })} aria-pressed={selectedTestament === 'new'}
            >
              {ko ? '신약' : 'New Testament'}
            </Button>
          </div>
        </div>

        {(selectedBook || searchQuery || selectedTestament !== 'all') && <div className="mb-5 flex flex-wrap items-center gap-3">
          <p role="status" className="text-sm text-muted-foreground">{selectedBook && `${selectedBook} · `}{resultCount} {ko ? '개의 묵상' : 'reflections'}</p>
          <Button variant="outline" size="sm" onClick={() => updateFilters({ q: '', book: '', testament: 'all', tab: 'bybook' })}>{ko ? '필터 초기화' : 'Clear filters'}</Button>
        </div>}
        <Tabs value={filters.tab} onValueChange={tab => updateFilters({ tab })} className="w-full">
          <TabsList className="mb-6">
            <TabsTrigger value="bybook">{ko ? '성경순' : 'By Bible book'}</TabsTrigger>
            <TabsTrigger value="recent">{ko ? '최근 묵상' : 'Recent reflections'}</TabsTrigger>
          </TabsList>

          {/* By Bible Book */}
          <TabsContent value="bybook">
            <MotionList className="grid gap-4">
              {filteredBooks.map((bookData) => (
                <MotionItem key={bookData.book}><Card className="overflow-hidden">
                  <CardHeader className="bg-gradient-to-r from-amber-50 to-transparent dark:from-amber-950/30 py-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex flex-wrap items-center gap-3">
                        <Badge
                          variant="outline"
                          className={bookData.bookInfo.testament === 'old'
                            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            : 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
                          }
                        >
                          {bookData.bookInfo.testament === 'old' ? (ko ? '구약' : 'Old Testament') : (ko ? '신약' : 'New Testament')}
                        </Badge>
                        <CardTitle className="text-lg">{bookData.book}</CardTitle>
                      </div>
                      <Badge variant="secondary">{bookData.entries.length}{ko ? '편' : ' reflections'}</Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="p-4">
                    <div className="grid gap-2">
                      {(selectedBook || query ? bookData.entries : bookData.entries.slice(0, 5)).map((entry) => (
                        <Link
                          key={entry.slug}
                          href={`/qt/${entry.slug}`}
                          className="flex items-center justify-between p-3 rounded-lg hover:bg-muted transition-colors group"
                        >
                          <div className="flex items-center gap-3 min-w-0">
                            <Book className="h-4 w-4 text-muted-foreground shrink-0" />
                            <div className="min-w-0">
                              <p className="font-medium truncate group-hover:text-primary">
                                <SourceText text={entry.title} />
                              </p>
                              {entry.bibleReference && (
                                <p className="text-xs text-muted-foreground">{entry.bibleReference}</p>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            <span className="text-xs text-muted-foreground">{entry.date}</span>
                            <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-primary" />
                          </div>
                        </Link>
                      ))}
                      {!selectedBook && !query && bookData.entries.length > 5 && (
                        <div className="text-center pt-2">
                          <Button variant="ghost" size="sm" asChild>
                            <Link href={`/qt/?book=${encodeURIComponent(bookData.book)}`} onClick={event => {
                              if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                              event.preventDefault(); updateFilters({ book: bookData.book, tab: 'bybook' });
                            }}>
                              {ko ? `+${bookData.entries.length - 5}개 더 보기` : `View ${bookData.entries.length - 5} more`}
                            </Link>
                          </Button>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card></MotionItem>
              ))}
            </MotionList>
          </TabsContent>

          {resultCount === 0 && <p role="status" className="py-8 text-center text-muted-foreground">{ko ? '검색 결과가 없습니다.' : 'No reflections found.'}</p>}
          {/* Recent Entries */}
          <TabsContent value="recent">
            <MotionList className="grid gap-3">
              {filteredRecent.map((entry) => (
                <MotionItem key={entry.slug}><Link data-motion-reveal
                  href={`/qt/${entry.slug}`}
                  className="block"
                >
                  <Card className="hover:shadow-md transition-shadow">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3 min-w-0">
                          <div className="p-2 bg-amber-100 dark:bg-amber-900 rounded-lg shrink-0">
                            <Calendar className="h-4 w-4 text-amber-700 dark:text-amber-300" />
                          </div>
                          <div className="min-w-0">
                            <p className="font-medium truncate"><SourceText text={entry.title} /></p>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <span>{entry.date}</span>
                              {entry.bibleReference && (
                                <>
                                  <span>·</span>
                                  <span>{entry.bibleReference}</span>
                                </>
                              )}
                            </div>
                          </div>
                        </div>
                        <ChevronRight className="h-5 w-5 text-muted-foreground shrink-0" />
                      </div>
                    </CardContent>
                  </Card>
                </Link></MotionItem>
              ))}
            </MotionList>
          </TabsContent>
        </Tabs>
      </section>

      {/* Footer */}
      <footer className="border-t py-8 mt-12">
        <div className="container text-center text-sm text-muted-foreground">
          <p>{stats.dateRange.start} ~ {stats.dateRange.end}</p>
          <p className="mt-1">{ko ? '매일 말씀 묵상과 함께하는 신앙 여정' : 'A journey of daily reflection'}</p>
        </div>
      </footer>
    </div>
  </LocalizedContent>;
}
