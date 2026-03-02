'use client';

import { useState, useMemo } from 'react';
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
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTestament, setSelectedTestament] = useState<'all' | 'old' | 'new'>('all');

  const filteredBooks = useMemo(() => {
    return byBook.filter(b => {
      const matchesTestament = selectedTestament === 'all' || b.bookInfo.testament === selectedTestament;
      const matchesSearch = searchQuery === '' ||
        b.book.includes(searchQuery) ||
        b.entries.some(e => e.title.toLowerCase().includes(searchQuery.toLowerCase()));
      return matchesTestament && matchesSearch;
    });
  }, [byBook, searchQuery, selectedTestament]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-amber-50/50 to-background dark:from-amber-950/20">
      {/* Header */}
      <header className="border-b bg-background/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="container py-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-amber-100 dark:bg-amber-900 rounded-lg">
              <BookOpen className="h-8 w-8 text-amber-700 dark:text-amber-300" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold">Quiet Time</h1>
              <p className="text-muted-foreground text-sm">성경과 함께하는 묵상 일지</p>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-2xl font-bold text-amber-600 dark:text-amber-400">{stats.total}</p>
                <p className="text-xs text-muted-foreground">총 묵상</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{stats.booksCount}</p>
                <p className="text-xs text-muted-foreground">성경 책</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">{stats.oldTestamentEntries}</p>
                <p className="text-xs text-muted-foreground">구약</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">{stats.newTestamentEntries}</p>
                <p className="text-xs text-muted-foreground">신약</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </header>

      <main className="container py-8">
        {/* Search and Filter */}
        <div className="flex flex-col md:flex-row gap-4 mb-8">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="묵상 제목 또는 성경 책 검색..."
              className="pl-10"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <div className="flex gap-2">
            <Button
              variant={selectedTestament === 'all' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedTestament('all')}
            >
              전체
            </Button>
            <Button
              variant={selectedTestament === 'old' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedTestament('old')}
            >
              구약
            </Button>
            <Button
              variant={selectedTestament === 'new' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedTestament('new')}
            >
              신약
            </Button>
          </div>
        </div>

        <Tabs defaultValue="bybook" className="w-full">
          <TabsList className="mb-6">
            <TabsTrigger value="bybook">성경순</TabsTrigger>
            <TabsTrigger value="recent">최근 묵상</TabsTrigger>
          </TabsList>

          {/* By Bible Book */}
          <TabsContent value="bybook">
            <div className="grid gap-4">
              {filteredBooks.map((bookData) => (
                <Card key={bookData.book} className="overflow-hidden">
                  <CardHeader className="bg-gradient-to-r from-amber-50 to-transparent dark:from-amber-950/30 py-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Badge
                          variant="outline"
                          className={bookData.bookInfo.testament === 'old'
                            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            : 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
                          }
                        >
                          {bookData.bookInfo.testament === 'old' ? '구약' : '신약'}
                        </Badge>
                        <CardTitle className="text-lg">{bookData.book}</CardTitle>
                      </div>
                      <Badge variant="secondary">{bookData.entries.length}편</Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="p-4">
                    <div className="grid gap-2">
                      {bookData.entries.slice(0, 5).map((entry) => (
                        <Link
                          key={entry.slug}
                          href={`/qt/${entry.slug}`}
                          className="flex items-center justify-between p-3 rounded-lg hover:bg-muted transition-colors group"
                        >
                          <div className="flex items-center gap-3 min-w-0">
                            <Book className="h-4 w-4 text-muted-foreground shrink-0" />
                            <div className="min-w-0">
                              <p className="font-medium truncate group-hover:text-primary">
                                {entry.title}
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
                      {bookData.entries.length > 5 && (
                        <div className="text-center pt-2">
                          <Button variant="ghost" size="sm" asChild>
                            <Link href={`/qt?book=${encodeURIComponent(bookData.book)}`}>
                              +{bookData.entries.length - 5}개 더 보기
                            </Link>
                          </Button>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Recent Entries */}
          <TabsContent value="recent">
            <div className="grid gap-3">
              {recentEntries.map((entry) => (
                <Link
                  key={entry.slug}
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
                            <p className="font-medium truncate">{entry.title}</p>
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
                </Link>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t py-8 mt-12">
        <div className="container text-center text-sm text-muted-foreground">
          <p>{stats.dateRange.start} ~ {stats.dateRange.end}</p>
          <p className="mt-1">매일 말씀 묵상과 함께하는 신앙 여정</p>
        </div>
      </footer>
    </div>
  );
}
