'use client';

import { useState } from 'react';
import { HomeIdentityHero } from '@/components/home-identity-hero';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight, Brain, Database, Eye, BarChart3, Network, MapPin, Youtube, BookOpen, Newspaper, FolderKanban, AudioLines, ExternalLink, Calendar, Wand2, PenLine } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useLanguage } from '@/components/language-provider';
import type { BlogPostMeta } from '@/lib/blog';
import type { Project } from '@/data/projects';
import type { Publication } from '@/data/publications';
import type { MediaArticle } from '@/data/media';

interface HomepageContentProps {
  recentPosts: BlogPostMeta[];
  selectedPublications: (Publication & { reasonKo: string; reasonEn: string })[];
  featuredProjects: Project[];
  stats: {
    publications: number;
    videos: number;
    projects: number;
    lectures: number;
  };
  mediaArticles: MediaArticle[];
}

const researchAreas = [
  { title: 'Data Science & Big Data', titleKo: '데이터과학 및 빅데이터', icon: Database, href: '/research/ds', color: 'from-blue-500 to-cyan-500' },
  { title: 'Deep Learning & ML', titleKo: '딥러닝 및 머신러닝', icon: Brain, href: '/research/dl', color: 'from-purple-500 to-pink-500' },
  { title: 'Natural Language Processing', titleKo: '자연어처리', icon: BookOpen, href: '/research/nlp', color: 'from-green-500 to-emerald-500' },
  { title: 'Computer Vision', titleKo: '컴퓨터 비전', icon: Eye, href: '/research/cv', color: 'from-orange-500 to-red-500' },
  { title: 'Graphs and Tensors', titleKo: '그래프 및 텐서', icon: Network, href: '/research/graphs', color: 'from-indigo-500 to-violet-500' },
  { title: 'Spatio-Temporal', titleKo: '시공간 데이터', icon: MapPin, href: '/research/st', color: 'from-teal-500 to-cyan-500' },
  { title: 'Audio & Speech Processing', titleKo: '오디오 음성 처리', icon: AudioLines, href: '/research/asp', color: 'from-rose-500 to-pink-500' },
];

const youtubeTopics = ['Python Programming', 'Data Science', 'Machine Learning', 'Deep Learning', 'Computer Vision', 'NLP'];

export function HomepageContent({
  recentPosts,
  selectedPublications,
  featuredProjects,
  stats,
  mediaArticles,
}: HomepageContentProps) {
  const { language, t } = useLanguage();
  const [showAllMedia, setShowAllMedia] = useState(false);

  const statsData = [
    { label: t('stats.publications') as string, value: stats.publications.toLocaleString() },
    { label: t('stats.videos') as string, value: stats.videos.toLocaleString() },
    { label: t('stats.projects') as string, value: stats.projects.toLocaleString() },
    { label: t('stats.lectures') as string, value: stats.lectures.toLocaleString() },
  ];

  return (
    <>
      <HomeIdentityHero stats={statsData} />

      <section className="border-b bg-muted/20 py-16 md:py-20" aria-labelledby="selected-research">
        <div className="container">
          <div className="mb-8 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
            <div>
              <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-primary">Selected Publications</p>
              <h2 id="selected-research" className="text-3xl font-semibold tracking-tight">{language === 'ko' ? '논문으로 만나는 SuanLab의 연구' : 'Research from SuanLab'}</h2>
              <p className="mt-3 text-muted-foreground">{language === 'ko' ? '신경망 방법론부터 실제 환경의 센싱과 생성 모델까지.' : 'From neural network methods to sensing and generation in real environments.'}</p>
            </div>
            <Link href="/publication" className="inline-flex shrink-0 items-center gap-2 text-sm font-medium text-primary hover:underline">{language === 'ko' ? '전체 연구 성과' : 'All publications'}<ArrowRight className="h-4 w-4" /></Link>
          </div>
          <div className="grid gap-5 lg:grid-cols-3">
            {selectedPublications.map((publication) => (
              <Card key={publication.id} className="relative flex h-full flex-col overflow-hidden border-t-2 border-t-primary/60">
                <CardHeader>
                  <p className="mb-3 text-xs font-medium text-primary">{publication.venue} · {publication.date}</p>
                  <CardTitle className="text-lg leading-relaxed"><a href={publication.url || '/publication/'} target={publication.url ? '_blank' : undefined} rel={publication.url ? 'noopener noreferrer' : undefined} className="hover:text-primary">{publication.title}</a></CardTitle>
                </CardHeader>
                <CardContent className="mt-auto"><p className="mb-4 text-sm leading-relaxed">{language === 'ko' ? publication.reasonKo : publication.reasonEn}</p><p className="text-sm leading-relaxed text-muted-foreground">{publication.authors}</p>{publication.badge && <Badge variant="secondary" className="mt-4">{publication.badge}</Badge>}</CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <section className="border-b py-16" aria-labelledby="current-projects">
        <div className="container">
          <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
            <div><p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-primary">Current Projects</p><h2 id="current-projects" className="text-3xl font-semibold">{language === 'ko' ? '진행 중인 연구 프로젝트' : 'Current research projects'}</h2></div>
            <Link href="/project/" className="text-sm text-primary hover:underline">{language === 'ko' ? '전체 프로젝트 →' : 'All projects →'}</Link>
          </div>
          <div className="grid gap-6 md:grid-cols-2">{featuredProjects.map(project => <Card key={project.id}>
            <CardHeader><p className="mb-3 text-xs text-muted-foreground">{project.organization} · {project.period}</p><CardTitle className="text-lg leading-relaxed"><Link href={`/project/#project-${project.id}`} className="hover:text-primary">{project.title}</Link></CardTitle></CardHeader>
            <CardContent><p className="mb-2 text-xs font-medium text-primary">{language === 'ko' ? '연구 목표' : 'Research objective'}</p><p className="text-sm leading-relaxed text-muted-foreground">{project.items[0]}</p></CardContent>
          </Card>)}</div>
        </div>
      </section>

      <section className="py-20 md:py-28">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <Badge variant="outline" className="mb-4">
              <Newspaper className="mr-2 h-3 w-3" />
              {t('media.badge') as string}
            </Badge>
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">{t('media.title') as string}</h2>
            <p className="mt-4 text-muted-foreground">
              {t('media.description') as string}
            </p>
          </div>

          <div id="media-articles" className="mt-12 grid gap-6 md:grid-cols-2 xl:grid-cols-4">
            {(showAllMedia ? mediaArticles : mediaArticles.slice(0, 4)).map((article) => (
              <a
                key={article.id}
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group"
              >
                <Card className="h-full transition-all hover:shadow-lg hover:-translate-y-1 hover:border-primary/50">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                      <Badge variant="secondary" className="text-xs font-normal">
                        {article.source}
                      </Badge>
                      <span className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {article.date}
                      </span>
                    </div>
                    <CardTitle className="text-base leading-tight line-clamp-2 group-hover:text-primary transition-colors">
                      {article.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <p className="text-sm text-muted-foreground line-clamp-3">
                      {article.excerpt}
                    </p>
                    <div className="mt-4 flex items-center text-xs text-primary font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                      Read more
                      <ExternalLink className="ml-1 h-3 w-3" />
                    </div>
                  </CardContent>
                </Card>
              </a>
            ))}
          </div>
          {mediaArticles.length > 4 && (
            <div className="mt-8 text-center">
              <Button variant="outline" aria-expanded={showAllMedia} aria-controls="media-articles" onClick={() => setShowAllMedia(!showAllMedia)}>
                {language === 'ko' ? (showAllMedia ? '보도 접기' : `전체 보도 ${mediaArticles.length}건 보기`) : (showAllMedia ? 'Show less' : `View all ${mediaArticles.length} articles`)}
              </Button>
            </div>
          )}
        </div>
      </section>

      <section className="bg-muted/30 py-20 md:py-28">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <Badge variant="outline" className="mb-4">{t('blog.badge') as string}</Badge>
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">{t('blog.title') as string}</h2>
            <p className="mt-4 text-muted-foreground">
              {t('blog.description') as string}
            </p>
          </div>

          <div className="mt-12 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {recentPosts.map((post) => (
              <Link key={post.slug} href={`/blog/${post.slug}`}>
                <Card className="group h-full transition-all hover:shadow-lg hover:-translate-y-1">
                  {post.thumbnail && (
                    <div className="aspect-video overflow-hidden">
                      <Image
                        src={post.thumbnail}
                        alt={post.title}
                        width={400}
                        height={225}
                        className="h-full w-full object-cover transition-transform group-hover:scale-105"
                      />
                    </div>
                  )}
                  <CardHeader>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
                      <Calendar className="h-3 w-3" />
                      <span>{post.date}</span>
                      <span className="text-muted-foreground/50">•</span>
                      <Badge variant="secondary" className="text-xs font-normal">
                        {post.category}
                      </Badge>
                    </div>
                    <CardTitle className="line-clamp-2 group-hover:text-primary transition-colors">
                      {post.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {post.excerpt}
                    </p>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>

          <div className="mt-10 text-center">
            <Button variant="outline" size="lg" asChild>
              <Link href="/blog">
                {t('blog.btn.more') as string} <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      <section className="py-20 md:py-28">
        <div className="container">
          <div className="grid gap-12 lg:grid-cols-2 lg:items-center">
            <div>
              <Badge className="mb-4">{t('youtube.badge') as string}</Badge>
              <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
                {t('youtube.title') as string}
              </h2>
              <p className="mt-4 text-muted-foreground">
                {t('youtube.description') as string}
              </p>
              <ul className="mt-8 space-y-3">
                {youtubeTopics.map((item) => (
                  <li key={item} className="flex items-center gap-3">
                    <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10">
                      <ArrowRight className="h-3 w-3 text-primary" />
                    </div>
                    {item}
                  </li>
                ))}
              </ul>
              <Button className="mt-8" size="lg" asChild>
                <Link href="/youtube">
                  {t('youtube.btn.watch') as string}
                  <Youtube className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </div>
            <div className="aspect-video overflow-hidden rounded-xl shadow-2xl">
              <iframe
                className="h-full w-full"
                loading="lazy"
                src="https://www.youtube.com/embed/k60oT_8lyFw"
                title="SuanLab YouTube"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
          </div>
        </div>
      </section>

      <section className="bg-muted/50 py-20 md:py-28">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">{t('quicklinks.title') as string}</h2>
            <p className="mt-4 text-muted-foreground">
              {t('quicklinks.description') as string}
            </p>
          </div>

          <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {[
              { title: 'Research', description: t('quicklinks.research') as string, icon: BarChart3, href: '/research' },
              { title: 'YouTube', description: t('quicklinks.youtube') as string, icon: Youtube, href: '/youtube' },
              { title: 'Publications', description: t('quicklinks.publications') as string, icon: Newspaper, href: '/publication' },
              { title: 'Projects', description: t('quicklinks.projects') as string, icon: FolderKanban, href: '/project' },
              { title: 'Prompts', description: t('quicklinks.prompts') as string, icon: Wand2, href: '/prompts' },
              { title: 'Blog', description: t('quicklinks.blog') as string, icon: PenLine, href: '/blog' },
            ].map((link) => {
              const Icon = link.icon;
              return (
                <Link key={link.href} href={link.href}>
                  <Card className="group h-full transition-all hover:shadow-lg hover:border-primary/50">
                    <CardContent className="flex flex-col items-center p-6 text-center">
                      <div className="mb-4 rounded-full bg-primary/10 p-4 group-hover:bg-primary/20 transition-colors">
                        <Icon className="h-8 w-8 text-primary" />
                      </div>
                      <h3 className="text-lg font-semibold">{link.title}</h3>
                      <p className="mt-2 text-sm text-muted-foreground">{link.description}</p>
                    </CardContent>
                  </Card>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      <section className="py-20 md:py-28">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">{t('research.title') as string}</h2>
            <p className="mt-4 text-muted-foreground">
              {t('research.description') as string}
            </p>
          </div>

          <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {researchAreas.map((area) => {
              const Icon = area.icon;
              return (
                <Link key={area.href} href={area.href}>
                  <Card className="group h-full transition-all hover:shadow-lg hover:-translate-y-1">
                    <CardHeader>
                      <div className={`mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br ${area.color}`}>
                        <Icon className="h-6 w-6 text-white" />
                      </div>
                      <CardTitle className="group-hover:text-primary transition-colors">
                        {area.title}
                      </CardTitle>
                      <CardDescription>{language === 'ko' ? area.titleKo : area.title}</CardDescription>
                    </CardHeader>
                  </Card>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-primary to-blue-600 py-20 text-white">
        <div className="container text-center">
          <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
            {t('cta.title') as string}
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-white/80">
            {t('cta.description') as string}
          </p>
          <div className="mt-10 flex flex-col gap-4 sm:flex-row sm:justify-center">
            <Button size="lg" variant="secondary" asChild>
              <Link href="/contact">{t('cta.btn.contact') as string}</Link>
            </Button>
            <Button size="lg" variant="outline" className="bg-transparent border-white text-white hover:bg-white/10" asChild>
              <Link href="/publication">{t('cta.btn.publications') as string}</Link>
            </Button>
          </div>
        </div>
      </section>
    </>
  );
}
