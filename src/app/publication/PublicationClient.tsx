'use client';

import { useLanguage } from '@/components/language-provider';
import { useState, useEffect } from 'react';
import {
  Users,
  Building,
  Calendar,
  ExternalLink,
  ChevronDown,
  Search,
  Copy,
  Check,
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import type { Publication, PublicationType } from '@/data/publications';

interface PublicationTypeOption {
  key: PublicationType | 'all';
  label: string;
  count: number;
}

interface PublicationEntry extends Publication {
  sortYear: number;
  sortMonth: number;
}

function getBibTeXType(type: Publication['type']): string {
  switch (type) {
    case 'journal':
    case 'djournal':
    case 'column':
      return 'article';
    case 'conference':
    case 'dconference':
      return 'inproceedings';
    case 'book':
      return 'book';
    case 'patent':
      return 'patent';
    case 'report':
      return 'techreport';
    default:
      return 'article';
  }
}

function generateBibTeX(pub: Publication, year: number): string {
  const key = `${pub.authors
    .split(',')[0]
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_')}_${year}`;
  const entryType = getBibTeXType(pub.type);

  if (entryType === 'inproceedings') {
    return `@inproceedings{${key},
  author = {${pub.authors}},
  title = {${pub.title}},
  booktitle = {${pub.venue}},
  year = {${year}}
}`;
  }

  if (entryType === 'book') {
    return `@book{${key},
  author = {${pub.authors}},
  title = {${pub.title}},
  publisher = {${pub.venue}},
  year = {${year}}
}`;
  }

  if (entryType === 'techreport') {
    return `@techreport{${key},
  author = {${pub.authors}},
  title = {${pub.title}},
  institution = {${pub.venue}},
  year = {${year}}
}`;
  }

  if (entryType === 'patent') {
    return `@patent{${key},
  author = {${pub.authors}},
  title = {${pub.title}},
  year = {${year}}
}`;
  }

  return `@article{${key},
  author = {${pub.authors}},
  title = {${pub.title}},
  journal = {${pub.venue}},
  year = {${year}}
}`;
}

interface PublicationClientProps {
  publications: PublicationEntry[];
  publicationTypes: PublicationTypeOption[];
}

export default function PublicationClient({
  publications,
  publicationTypes,
}: PublicationClientProps) {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [activeFilter, setActiveFilter] = useState<PublicationType | 'all'>(
    'all'
  );
  const [openToggle, setOpenToggle] = useState<number | null>(null);
  const [copiedId, setCopiedId] = useState<number | null>(null);

  const [query, setQuery] = useState('');
  const [year, setYear] = useState('all');
  useEffect(() => {
    setQuery(new URLSearchParams(window.location.search).get('q') || '');
    const match = window.location.hash.match(/^#publication-(\d+)$/);
    if (match) setOpenToggle(Number(match[1]));
  }, []);
  const years = Array.from(new Set(publications.map((p) => p.sortYear))).sort((a, b) => b - a);
  const filteredPublications = (
    activeFilter === 'all'
      ? publications
      : publications.filter((p) => p.type === activeFilter)
  ).filter((p) => (year === 'all' || String(p.sortYear) === year) &&
    `${p.title} ${p.authors} ${p.venue} ${p.keywords || ''}`.toLowerCase().includes(query.toLowerCase())
  ).sort((a, b) => {
    if (b.sortYear !== a.sortYear) return b.sortYear - a.sortYear;
    return b.sortMonth - a.sortMonth;
  });

  async function handleCopyBibTeX(pub: PublicationEntry) {
    const bibtex = generateBibTeX(pub, pub.sortYear);
    try {
      await navigator.clipboard.writeText(bibtex);
      setCopiedId(pub.id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy BibTeX:', err);
    }
  }

  return (
    <div className="lg:col-span-3">
      <div className="mb-6 grid gap-3 sm:grid-cols-[1fr_auto]">
        <label className="text-sm">검색 / Search<input className="mt-2 block w-full rounded-lg border bg-background p-3" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={ko ? '제목, 저자, 학술지, 키워드' : 'Title, author, venue, keywords'} /></label>
        <label className="text-sm">연도 / Year<select className="mt-2 block w-full rounded-lg border bg-background p-3" value={year} onChange={(event) => setYear(event.target.value)}><option value="all">전체 / All</option>{years.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
      </div>
      <p className="mb-4 text-sm text-muted-foreground" aria-live="polite">{filteredPublications.length} {ko ? '개 결과' : 'results'}</p>
      <div className="mb-8 flex flex-wrap gap-2">
        {publicationTypes.map((type) => (
          <Button
            key={type.key}
            variant={activeFilter === type.key ? 'default' : 'outline'}
            size="sm"
            onClick={() => setActiveFilter(type.key)}
            aria-pressed={activeFilter === type.key}
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

      <div className="space-y-4">
        {filteredPublications.map((pub) => (
          <Card
            key={pub.id}
            id={`publication-${pub.id}`}
            className={`transition-all ${
              openToggle === pub.id ? 'ring-2 ring-primary' : ''
            }`}
          >
            <CardContent className="p-4">
              <button
                onClick={() =>
                  setOpenToggle(openToggle === pub.id ? null : pub.id)
                }
                className="w-full text-left"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex flex-wrap items-center gap-2 mb-2">
                      {pub.badge && (
                        <Badge
                          className={cn(
                            'text-xs',
                            pub.badge === 'SCIE'
                              ? 'bg-purple-700 text-white hover:bg-purple-600'
                              : pub.badge === 'SSCI'
                              ? 'bg-yellow-800 text-white hover:bg-yellow-600'
                              : pub.badge === 'KCI'
                              ? 'bg-blue-700 text-white hover:bg-blue-600'
                              : pub.badge.includes('장려상')
                              ? 'bg-emerald-700 text-white hover:bg-emerald-700'
                              : pub.badge.includes('우수')
                              ? 'bg-amber-800 text-white hover:bg-amber-600'
                              : pub.badge.includes('Best Paper Award Runner-up')
                              ? 'bg-orange-500 text-white hover:bg-orange-600'
                              : 'bg-destructive text-destructive-foreground'
                          )}
                        >
                          {pub.badge}
                        </Badge>
                      )}
                      {pub.impact && (
                        <Badge
                          variant="outline"
                          className="text-xs text-orange-700 dark:text-orange-400 border-orange-600"
                        >
                          {pub.impact}
                        </Badge>
                      )}
                    </div>
                    <h3 className="font-medium leading-relaxed">
                      <span
                        dangerouslySetInnerHTML={{
                          __html: pub.authors
                            .replace(
                              /Suan Lee/g,
                              '<strong class="text-primary">Suan Lee</strong>'
                            )
                            .replace(
                              /이수안/g,
                              '<strong class="text-primary">이수안</strong>'
                            ),
                        }}
                      />
                      , &ldquo;<strong>{pub.title}</strong>&rdquo; {pub.venue},{' '}
                      ({pub.date}).
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

                  <div className="flex gap-2">
                    {pub.url && (
                      <Button variant="outline" size="sm" asChild>
                        <a
                          href={pub.url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <ExternalLink className="mr-2 h-4 w-4" />
                          Read More
                        </a>
                      </Button>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleCopyBibTeX(pub)}
                    >
                      {copiedId === pub.id ? (
                        <>
                          <Check className="mr-2 h-4 w-4" />
                          {ko ? '복사됨' : 'Copied'}!
                        </>
                      ) : (
                        <>
                          <Copy className="mr-2 h-4 w-4" />
                          {ko ? 'BibTeX 복사' : 'Copy BibTeX'}
                        </>
                      )}
                    </Button>
                  </div>
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
            <h3 className="text-lg font-semibold">{ko ? '해당하는 논문이 없습니다.' : 'No publications found'}</h3>
            <p className="mt-2 text-muted-foreground">
              해당 카테고리에 출판물이 없습니다.
            </p>
          </div>
        </Card>
      )}
    </div>
  );
}
