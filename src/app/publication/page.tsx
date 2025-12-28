'use client';

import { useState } from 'react';
import { FileText, Users, Building, Calendar, ExternalLink, ChevronDown, Search, GraduationCap, BookOpen } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { publications, publicationTypes, PublicationType, Publication } from '@/data/publications';

// 날짜 문자열에서 연도를 추출하는 함수
function extractYear(dateStr: string): number {
  // "2025", "2024년", "January 2024", "December 2020", "2024년 12월" 등 다양한 형식 처리
  const yearMatch = dateStr.match(/\b(19|20)\d{2}\b/);
  if (yearMatch) {
    return parseInt(yearMatch[0], 10);
  }
  return 0;
}

// 날짜 문자열에서 월을 추출하는 함수
function extractMonth(dateStr: string): number {
  const monthNames: { [key: string]: number } = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
  };

  const lowerDate = dateStr.toLowerCase();
  for (const [month, num] of Object.entries(monthNames)) {
    if (lowerDate.includes(month)) {
      return num;
    }
  }

  // 한국어 월 처리 ("12월", "11월" 등)
  const koreanMonthMatch = dateStr.match(/(\d{1,2})월/);
  if (koreanMonthMatch) {
    return parseInt(koreanMonthMatch[1], 10);
  }

  return 0;
}

// 날짜순 정렬 함수 (최신순)
function sortByDate(a: Publication, b: Publication): number {
  const yearA = extractYear(a.date);
  const yearB = extractYear(b.date);

  if (yearA !== yearB) {
    return yearB - yearA; // 최신 연도가 먼저
  }

  const monthA = extractMonth(a.date);
  const monthB = extractMonth(b.date);

  return monthB - monthA; // 최신 월이 먼저
}

export default function PublicationPage() {
  const [activeFilter, setActiveFilter] = useState<PublicationType | 'all'>('all');
  const [openToggle, setOpenToggle] = useState<number | null>(null);

  const filteredPublications = (activeFilter === 'all'
    ? publications
    : publications.filter((p) => p.type === activeFilter)
  ).sort(sortByDate);

  return (
    <>
      <PageHeader
        title="Publication"
        subtitle="Research publications including journals, conferences, and more"
        breadcrumbs={[{ label: 'Publication' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="grid gap-8 lg:grid-cols-4">
            {/* Sidebar */}
            <aside className="lg:col-span-1">
              <div className="sticky top-24">
                {/* External Links */}
                <div className="mb-6 flex flex-col gap-2">
                  <Button variant="default" className="w-full justify-start" asChild>
                    <a
                      href="https://scholar.google.com/citations?user=mK5U7hgAAAAJ&hl=en"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <GraduationCap className="mr-2 h-4 w-4" />
                      Google Scholar
                    </a>
                  </Button>
                  <Button variant="secondary" className="w-full justify-start" asChild>
                    <a
                      href="http://dblp.uni-trier.de/pers/hd/l/Lee:Suan"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <BookOpen className="mr-2 h-4 w-4" />
                      DBLP
                    </a>
                  </Button>
                </div>

                <div className="mb-6">
                  <div className="flex items-center gap-2 mb-4">
                    <FileText className="h-5 w-5 text-primary" />
                    <h2 className="font-semibold">Statistics</h2>
                  </div>
                  <Card>
                    <CardContent className="p-4 space-y-3">
                      {publicationTypes.slice(1).map((type) => (
                        <div key={type.key} className="flex justify-between text-sm">
                          <span className="text-muted-foreground">{type.label}</span>
                          <Badge variant="secondary">{type.count}</Badge>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                </div>
              </div>
            </aside>

            {/* Main Content */}
            <div className="lg:col-span-3">
              {/* Filter */}
              <div className="mb-8 flex flex-wrap gap-2">
                {publicationTypes.map((type) => (
                  <Button
                    key={type.key}
                    variant={activeFilter === type.key ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setActiveFilter(type.key)}
                    className="relative"
                  >
                    {type.label}
                    <Badge
                      variant={activeFilter === type.key ? 'secondary' : 'outline'}
                      className="ml-2 text-xs"
                    >
                      {type.count}
                    </Badge>
                  </Button>
                ))}
              </div>

              {/* Publications List */}
              <div className="space-y-4">
                {filteredPublications.map((pub) => (
                  <Card
                    key={pub.id}
                    className={`transition-all ${openToggle === pub.id ? 'ring-2 ring-primary' : ''}`}
                  >
                    <CardContent className="p-4">
                      <button
                        onClick={() => setOpenToggle(openToggle === pub.id ? null : pub.id)}
                        className="w-full text-left"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1">
                            <div className="flex flex-wrap items-center gap-2 mb-2">
                              {pub.badge && (
                                <Badge variant="destructive" className="text-xs">
                                  {pub.badge}
                                </Badge>
                              )}
                              {pub.impact && (
                                <Badge variant="outline" className="text-xs text-orange-600 border-orange-600">
                                  {pub.impact}
                                </Badge>
                              )}
                            </div>
                            <h3 className="font-medium leading-relaxed">
                              <span
                                dangerouslySetInnerHTML={{
                                  __html: pub.authors
                                    .replace(/Suan Lee/g, '<strong class="text-primary">Suan Lee</strong>')
                                    .replace(/이수안/g, '<strong class="text-primary">이수안</strong>'),
                                }}
                              />
                              , &ldquo;<strong>{pub.title}</strong>&rdquo; {pub.venue}, ({pub.date}).
                            </h3>
                          </div>
                          <ChevronDown
                            className={`h-5 w-5 text-muted-foreground transition-transform ${
                              openToggle === pub.id ? 'rotate-180' : ''
                            }`}
                          />
                        </div>
                      </button>

                      {openToggle === pub.id && (
                        <div className="mt-4 pt-4 border-t space-y-4">
                          <div className="grid gap-3 text-sm">
                            <div className="flex items-center gap-2 text-muted-foreground">
                              <Users className="h-4 w-4" />
                              <span>{pub.authors}</span>
                            </div>
                            <div className="flex items-center gap-2 text-muted-foreground">
                              <Building className="h-4 w-4" />
                              <span>{pub.venue}</span>
                            </div>
                            <div className="flex items-center gap-2 text-muted-foreground">
                              <Calendar className="h-4 w-4" />
                              <span>{pub.date}</span>
                            </div>
                          </div>

                          {pub.abstract && (
                            <div>
                              <h4 className="font-medium mb-2">Abstract</h4>
                              <p className="text-sm text-muted-foreground leading-relaxed">
                                {pub.abstract}
                              </p>
                            </div>
                          )}

                          {pub.keywords && (
                            <div>
                              <h4 className="font-medium mb-2">Keywords</h4>
                              <div className="flex flex-wrap gap-2">
                                {pub.keywords.split(';').map((keyword, idx) => (
                                  <Badge key={idx} variant="secondary" className="text-xs">
                                    {keyword.trim()}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}

                          {pub.url && (
                            <Button variant="outline" size="sm" asChild>
                              <a href={pub.url} target="_blank" rel="noopener noreferrer">
                                <ExternalLink className="mr-2 h-4 w-4" />
                                Read More
                              </a>
                            </Button>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>

              {filteredPublications.length === 0 && (
                <Card className="p-12">
                  <div className="flex flex-col items-center text-center">
                    <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                      <Search className="h-8 w-8 text-muted-foreground" />
                    </div>
                    <h3 className="text-lg font-semibold">No publications found</h3>
                    <p className="mt-2 text-muted-foreground">
                      해당 카테고리에 출판물이 없습니다.
                    </p>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
