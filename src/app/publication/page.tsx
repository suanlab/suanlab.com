import type { Metadata } from 'next';
import { FileText, GraduationCap, BookOpen, ExternalLink } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { ScholarlyArticleCollectionJsonLd } from '@/components/seo/JsonLd';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  publications,
  publicationTypes,
  type Publication,
} from '@/data/publications';
import PublicationClient from './PublicationClient';

export const metadata: Metadata = {
  title: 'Publications | SuanLab',
  description:
    '이수안 교수의 학술 논문 및 연구 성과 - SCI/SCIE 국제 저널, 국내 학술지, 학술대회 논문 목록',
  keywords: [
    '논문',
    'Publications',
    'SCI',
    'SCIE',
    '학술논문',
    '연구성과',
    '이수안',
    'Deep Learning',
    'Machine Learning',
  ],
  openGraph: {
    title: 'Publications | SuanLab',
    description: '이수안 교수의 학술 논문 및 연구 성과',
    url: 'https://suanlab.com/publication',
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Publications | SuanLab',
    description: '이수안 교수의 학술 논문 및 연구 성과',
  },
  alternates: {
    canonical: 'https://suanlab.com/publication',
  },
};

function extractYear(dateStr: string): number {
  const yearMatch = dateStr.match(/\b(19|20)\d{2}\b/);
  if (yearMatch) {
    return parseInt(yearMatch[0], 10);
  }
  return 0;
}

function extractMonth(dateStr: string): number {
  const monthNames: Record<string, number> = {
    january: 1,
    february: 2,
    march: 3,
    april: 4,
    may: 5,
    june: 6,
    july: 7,
    august: 8,
    september: 9,
    october: 10,
    november: 11,
    december: 12,
  };

  const lowerDate = dateStr.toLowerCase();
  for (const [month, num] of Object.entries(monthNames)) {
    if (lowerDate.includes(month)) {
      return num;
    }
  }

  const koreanMonthMatch = dateStr.match(/(\d{1,2})월/);
  if (koreanMonthMatch) {
    return parseInt(koreanMonthMatch[1], 10);
  }

  return 0;
}

interface PublicationEntry extends Publication {
  sortYear: number;
  sortMonth: number;
}

function preparePublications(pubs: Publication[]): PublicationEntry[] {
  return pubs
    .map((pub) => ({
      ...pub,
      sortYear: extractYear(pub.date),
      sortMonth: extractMonth(pub.date),
    }))
    .sort((a, b) => {
      if (b.sortYear !== a.sortYear) return b.sortYear - a.sortYear;
      return b.sortMonth - a.sortMonth;
    });
}

export default function PublicationPage() {
  const sortedPublications = preparePublications(publications);

  return (
    <>
      <ScholarlyArticleCollectionJsonLd publications={publications.map((p) => ({
        title: p.title,
        authors: p.authors,
        venue: p.venue,
        date: p.date,
        url: p.url,
      }))} />
      <PageHeader
        title="Publication"
        subtitle="Research publications including journals, conferences, and more"
        breadcrumbs={[{ label: 'Publication' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="grid gap-8 lg:grid-cols-4">
            <aside className="lg:col-span-1">
              <div className="sticky top-24">
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
                  <Button variant="secondary" className="w-full justify-start" asChild>
                    <a
                      href="https://orcid.org/0000-0002-3047-1167"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <ExternalLink className="mr-2 h-4 w-4" />
                      ORCID
                    </a>
                  </Button>
                  <Button variant="secondary" className="w-full justify-start" asChild>
                    <a
                      href="https://www.scopus.com/authid/detail.uri?authorId=56023436400"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <ExternalLink className="mr-2 h-4 w-4" />
                      Scopus
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

            <PublicationClient
              publications={sortedPublications}
              publicationTypes={publicationTypes.map((t) => ({
                key: t.key,
                label: t.label,
                count: t.count,
              }))}
            />
          </div>
        </div>
      </section>
    </>
  );
}
