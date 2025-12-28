import Image from 'next/image';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { Database, Brain, BookOpen, Eye, Network, MapPin, ArrowRight, Check, Lightbulb, Award, GraduationCap, FolderOpen, Mic, FileText, ExternalLink } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { researchAreas, getResearchBySlug, researchKeywordMap } from '@/data/research';
import { publications, Publication } from '@/data/publications';

// 연구 분야별 관련 논문 필터링 함수
function getRelatedPublications(slug: string): Publication[] {
  const keywords = researchKeywordMap[slug] || [];
  if (keywords.length === 0) return [];

  const matchedPubs = publications.filter((pub) => {
    const searchText = `${pub.title} ${pub.keywords || ''} ${pub.venue}`.toLowerCase();
    return keywords.some((kw) => searchText.includes(kw.toLowerCase()));
  });

  // 제목 기준 중복 제거 (원본 데이터에 동일 논문이 다른 ID로 등록된 경우 처리)
  const seenTitles = new Set<string>();
  const uniquePubs = matchedPubs.filter((pub) => {
    const normalizedTitle = pub.title.toLowerCase().trim();
    if (seenTitles.has(normalizedTitle)) {
      return false;
    }
    seenTitles.add(normalizedTitle);
    return true;
  });

  // 최신순 정렬
  return uniquePubs.sort((a, b) => {
    const yearA = parseInt(a.date.match(/\d{4}/)?.[0] || '0');
    const yearB = parseInt(b.date.match(/\d{4}/)?.[0] || '0');
    return yearB - yearA;
  });
}

const iconMap: Record<string, React.ElementType> = {
  'et-gears': Database,
  'et-layers': Brain,
  'et-document': BookOpen,
  'et-pictures': Eye,
  'et-linegraph': Network,
  'et-map': MapPin,
  'et-music': Mic,
};

const colorMap: Record<string, string> = {
  ds: 'from-blue-500 to-cyan-500',
  dl: 'from-purple-500 to-pink-500',
  nlp: 'from-green-500 to-emerald-500',
  cv: 'from-orange-500 to-red-500',
  graphs: 'from-indigo-500 to-violet-500',
  st: 'from-teal-500 to-cyan-500',
  asp: 'from-rose-500 to-pink-500',
};

const bgColorMap: Record<string, string> = {
  ds: 'bg-blue-50 dark:bg-blue-950/30',
  dl: 'bg-purple-50 dark:bg-purple-950/30',
  nlp: 'bg-green-50 dark:bg-green-950/30',
  cv: 'bg-orange-50 dark:bg-orange-950/30',
  graphs: 'bg-indigo-50 dark:bg-indigo-950/30',
  st: 'bg-teal-50 dark:bg-teal-950/30',
  asp: 'bg-rose-50 dark:bg-rose-950/30',
};

export async function generateStaticParams() {
  return researchAreas.map((area) => ({
    slug: area.slug,
  }));
}

export async function generateMetadata({ params }: { params: { slug: string } }) {
  const area = getResearchBySlug(params.slug);
  if (!area) return { title: 'Not Found' };

  return {
    title: `${area.titleEn} | Research | SuanLab`,
    description: area.description,
  };
}

export default function ResearchDetailPage({ params }: { params: { slug: string } }) {
  const area = getResearchBySlug(params.slug);

  if (!area) {
    notFound();
  }

  const Icon = iconMap[area.icon] || Database;
  const color = colorMap[area.slug] || 'from-blue-500 to-cyan-500';
  const bgColor = bgColorMap[area.slug] || 'bg-blue-50 dark:bg-blue-950/30';

  return (
    <>
      <PageHeader
        title={area.titleEn}
        subtitle={area.titleKo}
        breadcrumbs={[
          { label: 'Research', href: '/research' },
          { label: area.titleEn },
        ]}
      />

      {/* Hero Section */}
      <section className="py-16 md:py-20">
        <div className="container">
          <div className="grid gap-12 lg:grid-cols-2 lg:items-start">
            <div className="relative aspect-video overflow-hidden rounded-xl shadow-2xl lg:sticky lg:top-24">
              <Image
                src={area.image}
                alt={area.titleEn}
                fill
                className="object-cover"
              />
              <div className={`absolute inset-0 bg-gradient-to-t ${color} opacity-20`} />
            </div>
            <div>
              <div className={`mb-6 inline-flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br ${color}`}>
                <Icon className="h-7 w-7 text-white" />
              </div>
              <h2 className="text-2xl font-bold md:text-3xl">{area.titleEn}</h2>
              <p className="mt-2 text-lg text-muted-foreground">{area.titleKo}</p>

              {/* Overview */}
              <div className="mt-6 space-y-4">
                {area.overview.split('\n\n').map((paragraph, idx) => (
                  <p key={idx} className="text-muted-foreground leading-relaxed">
                    {paragraph.trim()}
                  </p>
                ))}
              </div>

              {/* Keywords */}
              <div className="mt-6 flex flex-wrap gap-2">
                {area.keywords.map((keyword) => (
                  <Badge key={keyword} variant="secondary" className="text-xs">
                    {keyword}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Key Technologies */}
      <section className={`py-16 md:py-20 ${bgColor}`}>
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-12">
            <h2 className="text-2xl font-bold md:text-3xl">
              Key <span className="text-primary">Technologies</span>
            </h2>
            <p className="mt-4 text-muted-foreground">
              본 연구 분야에서 활용하는 핵심 기술 스택
            </p>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {area.keyTechnologies.map((tech) => (
              <Card key={tech} className="group hover:shadow-md transition-shadow">
                <CardContent className="flex items-center gap-3 p-4">
                  <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br ${color}`}>
                    <Check className="h-4 w-4 text-white" />
                  </div>
                  <span className="text-sm font-medium">{tech}</span>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Research Topics */}
      <section className="py-16 md:py-20">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-12">
            <h2 className="text-2xl font-bold md:text-3xl">
              Research <span className="text-primary">Topics</span>
            </h2>
            <p className="mt-4 text-muted-foreground">
              주요 연구 주제 및 세부 연구 분야
            </p>
          </div>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {area.researchTopics.map((topic, idx) => (
              <Card key={idx} className="group hover:shadow-lg transition-all hover:border-primary/50">
                <CardHeader className="pb-3">
                  <div className="flex items-start gap-4">
                    <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br ${color} text-white font-bold`}>
                      {String(idx + 1).padStart(2, '0')}
                    </div>
                    <CardTitle className="text-lg group-hover:text-primary transition-colors">
                      {topic.title}
                    </CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {topic.description}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Achievements & Projects */}
      <section className={`py-16 md:py-20 ${bgColor}`}>
        <div className="container">
          <div className="grid gap-8 lg:grid-cols-2">
            {/* Achievements */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Award className="h-5 w-5 text-primary" />
                  주요 성과
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {area.achievements.map((achievement, idx) => (
                    <li key={idx} className="flex items-start gap-3">
                      <div className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-gradient-to-br ${color} mt-0.5`}>
                        <Check className="h-3.5 w-3.5 text-white" />
                      </div>
                      <span className="text-sm leading-relaxed">{achievement}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Representative Projects */}
            {area.representativeProjects && area.representativeProjects.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FolderOpen className="h-5 w-5 text-primary" />
                    대표 연구과제
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-3">
                    {area.representativeProjects.map((project, idx) => (
                      <li key={idx} className="flex items-start gap-3">
                        <div className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-gradient-to-br ${color} mt-0.5`}>
                          <Lightbulb className="h-3.5 w-3.5 text-white" />
                        </div>
                        <span className="text-sm leading-relaxed">{project}</span>
                      </li>
                    ))}
                  </ul>
                  <div className="mt-6">
                    <Button variant="outline" size="sm" asChild>
                      <Link href="/project">
                        전체 연구과제 보기
                        <ArrowRight className="ml-2 h-4 w-4" />
                      </Link>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </section>

      {/* Related Publications */}
      {(() => {
        const relatedPubs = getRelatedPublications(area.slug);
        const journalPubs = relatedPubs.filter(p => p.type === 'journal');
        const confPubs = relatedPubs.filter(p => p.type === 'conference');
        const domesticJournalPubs = relatedPubs.filter(p => p.type === 'djournal');
        const domesticConfPubs = relatedPubs.filter(p => p.type === 'dconference');

        if (relatedPubs.length === 0) return null;

        return (
          <section className="py-16 md:py-20">
            <div className="container">
              <div className="mx-auto max-w-2xl text-center mb-12">
                <h2 className="text-2xl font-bold md:text-3xl">
                  Related <span className="text-primary">Publications</span>
                </h2>
                <p className="mt-4 text-muted-foreground">
                  {area.titleKo} 분야와 관련된 논문 총 {relatedPubs.length}편
                </p>
              </div>

              <div className="space-y-8">
                {/* International Journals */}
                {journalPubs.length > 0 && (
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <FileText className="h-5 w-5 text-primary" />
                        International Journals
                        <Badge variant="secondary" className="ml-2">{journalPubs.length}</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="max-h-80 overflow-y-auto pr-2 space-y-3">
                        {journalPubs.map((pub) => (
                          <div key={pub.id} className="pb-3 border-b border-muted last:border-0">
                            <div className="flex items-start gap-2">
                              {pub.badge && (
                                <Badge variant="outline" className="shrink-0 text-xs">{pub.badge}</Badge>
                              )}
                              <div className="flex-1 min-w-0">
                                <p className="font-medium text-sm leading-relaxed">
                                  {pub.url ? (
                                    <a href={pub.url} target="_blank" rel="noopener noreferrer" className="hover:text-primary hover:underline">
                                      {pub.title}
                                    </a>
                                  ) : pub.title}
                                </p>
                                <p className="text-xs text-muted-foreground mt-1">{pub.authors}</p>
                                <p className="text-xs text-muted-foreground">
                                  {pub.venue}, {pub.date} {pub.impact && <span className="text-primary">{pub.impact}</span>}
                                </p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* International Conferences */}
                {confPubs.length > 0 && (
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <FileText className="h-5 w-5 text-primary" />
                        International Conferences
                        <Badge variant="secondary" className="ml-2">{confPubs.length}</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="max-h-80 overflow-y-auto pr-2 space-y-3">
                        {confPubs.map((pub) => (
                          <div key={pub.id} className="pb-3 border-b border-muted last:border-0">
                            <div className="flex items-start gap-2">
                              {pub.badge && (
                                <Badge variant="outline" className="shrink-0 text-xs">{pub.badge}</Badge>
                              )}
                              <div className="flex-1 min-w-0">
                                <p className="font-medium text-sm leading-relaxed">
                                  {pub.url ? (
                                    <a href={pub.url} target="_blank" rel="noopener noreferrer" className="hover:text-primary hover:underline">
                                      {pub.title}
                                    </a>
                                  ) : pub.title}
                                </p>
                                <p className="text-xs text-muted-foreground mt-1">{pub.authors}</p>
                                <p className="text-xs text-muted-foreground">{pub.venue}, {pub.date}</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Domestic Journals */}
                {domesticJournalPubs.length > 0 && (
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <FileText className="h-5 w-5 text-primary" />
                        국내 학술지
                        <Badge variant="secondary" className="ml-2">{domesticJournalPubs.length}</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="max-h-60 overflow-y-auto pr-2 space-y-3">
                        {domesticJournalPubs.map((pub) => (
                          <div key={pub.id} className="pb-3 border-b border-muted last:border-0">
                            <div className="flex-1 min-w-0">
                              <p className="font-medium text-sm leading-relaxed">{pub.title}</p>
                              <p className="text-xs text-muted-foreground mt-1">{pub.authors}</p>
                              <p className="text-xs text-muted-foreground">{pub.venue}, {pub.date}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Domestic Conferences */}
                {domesticConfPubs.length > 0 && (
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <FileText className="h-5 w-5 text-primary" />
                        국내 학술대회
                        <Badge variant="secondary" className="ml-2">{domesticConfPubs.length}</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="max-h-60 overflow-y-auto pr-2 space-y-3">
                        {domesticConfPubs.map((pub) => (
                          <div key={pub.id} className="pb-3 border-b border-muted last:border-0">
                            <div className="flex-1 min-w-0">
                              <p className="font-medium text-sm leading-relaxed">{pub.title}</p>
                              <p className="text-xs text-muted-foreground mt-1">{pub.authors}</p>
                              <p className="text-xs text-muted-foreground">{pub.venue}, {pub.date}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>

              <div className="mt-8 text-center">
                <Button variant="outline" asChild>
                  <Link href="/publication">
                    전체 논문 보기
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </div>
            </div>
          </section>
        );
      })()}

      {/* Related Courses */}
      {area.relatedCourses && area.relatedCourses.length > 0 && (
        <section className={`py-16 md:py-20 ${bgColor}`}>
          <div className="container">
            <div className="mx-auto max-w-2xl text-center mb-12">
              <h2 className="text-2xl font-bold md:text-3xl">
                Related <span className="text-primary">Courses</span>
              </h2>
              <p className="mt-4 text-muted-foreground">
                이 연구 분야와 관련된 강의
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-4">
              {area.relatedCourses.map((course) => (
                <Link key={course} href="/lecture">
                  <Card className="group hover:shadow-md hover:border-primary/50 transition-all">
                    <CardContent className="flex items-center gap-3 p-4">
                      <GraduationCap className={`h-5 w-5 text-primary`} />
                      <span className="font-medium group-hover:text-primary transition-colors">
                        {course}
                      </span>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
            <div className="mt-8 text-center">
              <Button variant="outline" asChild>
                <Link href="/lecture">
                  전체 강의 보기
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
        </section>
      )}

      {/* Other Research Areas */}
      <section className="py-16 md:py-20">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-12">
            <h2 className="text-2xl font-bold md:text-3xl">
              Other Research <span className="text-primary">Areas</span>
            </h2>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {researchAreas
              .filter((r) => r.slug !== area.slug)
              .map((r) => {
                const RIcon = iconMap[r.icon] || Database;
                const rColor = colorMap[r.slug] || 'from-blue-500 to-cyan-500';
                return (
                  <Link key={r.slug} href={`/research/${r.slug}`}>
                    <Card className="group transition-all hover:shadow-md hover:border-primary/50 h-full">
                      <CardContent className="flex items-center gap-4 p-4">
                        <div className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br ${rColor}`}>
                          <RIcon className="h-6 w-6 text-white" />
                        </div>
                        <div className="min-w-0">
                          <h3 className="font-semibold group-hover:text-primary transition-colors truncate">
                            {r.titleEn}
                          </h3>
                          <p className="text-sm text-muted-foreground">{r.titleKo}</p>
                        </div>
                      </CardContent>
                    </Card>
                  </Link>
                );
              })}
          </div>
        </div>
      </section>
    </>
  );
}
