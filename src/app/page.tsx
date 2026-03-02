import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight, Brain, Database, Eye, BarChart3, Network, MapPin, Youtube, BookOpen, Newspaper, FolderKanban, AudioLines, ExternalLink, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { mediaArticles } from '@/data/media';
import { publications } from '@/data/publications';
import { projects } from '@/data/projects';
import { lectures } from '@/data/lectures';
import { playlists } from '@/data/youtube';
import { getAllPosts } from '@/lib/blog';

const researchAreas = [
  { title: 'Data Science & Big Data', titleKo: '데이터과학 및 빅데이터', icon: Database, href: '/research/ds', color: 'from-blue-500 to-cyan-500' },
  { title: 'Deep Learning & ML', titleKo: '딥러닝 및 머신러닝', icon: Brain, href: '/research/dl', color: 'from-purple-500 to-pink-500' },
  { title: 'Natural Language Processing', titleKo: '자연어처리', icon: BookOpen, href: '/research/nlp', color: 'from-green-500 to-emerald-500' },
  { title: 'Computer Vision', titleKo: '컴퓨터 비전', icon: Eye, href: '/research/cv', color: 'from-orange-500 to-red-500' },
  { title: 'Graphs and Tensors', titleKo: '그래프 및 텐서', icon: Network, href: '/research/graphs', color: 'from-indigo-500 to-violet-500' },
  { title: 'Spatio-Temporal', titleKo: '시공간 데이터', icon: MapPin, href: '/research/st', color: 'from-teal-500 to-cyan-500' },
  { title: 'Audio & Speech Processing', titleKo: '오디오 음성 처리', icon: AudioLines, href: '/research/asp', color: 'from-rose-500 to-pink-500' },
];

// Calculate stats from data
const totalVideos = playlists.reduce((acc, playlist) => acc + playlist.videos.length, 0);

const stats = [
  { label: '논문', value: `${publications.length}+` },
  { label: 'YouTube 영상', value: `${totalVideos}+` },
  { label: '프로젝트', value: `${projects.length}+` },
  { label: '강의', value: `${lectures.length}+` },
];

const quickLinks = [
  { title: 'Research', description: '연구 분야 탐색', icon: BarChart3, href: '/research' },
  { title: 'YouTube', description: '교육 영상 보기', icon: Youtube, href: '/youtube' },
  { title: 'Publications', description: '논문 목록', icon: Newspaper, href: '/publication' },
  { title: 'Projects', description: '프로젝트 목록', icon: FolderKanban, href: '/project' },
];

export default async function Home() {
  const recentPosts = getAllPosts().slice(0, 6);

  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
        <div className="absolute inset-0 bg-[url('/assets/images/slider/2.jpg')] bg-cover bg-center opacity-20" />
        <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-transparent" />

        <div className="container relative py-24 md:py-32 lg:py-40">
          <div className="mx-auto max-w-3xl text-center">
            <Badge variant="secondary" className="mb-4">
              데이터 사이언스 & 인공지능 연구실
            </Badge>
            <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
              <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                SuanLab
              </span>에 오신 것을 환영합니다
            </h1>
            <p className="mt-6 text-lg text-slate-300 md:text-xl">
              데이터 사이언스, 딥러닝, 머신러닝, 빅데이터 연구실입니다. 논문, 강의, YouTube 콘텐츠를 통해 지식을 공유합니다.
            </p>
            <div className="mt-10 flex flex-col gap-4 sm:flex-row sm:justify-center">
              <Button size="lg" asChild>
                <Link href="/suan">
                  이수안 교수 소개
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" className="bg-white/10 hover:bg-white/20" asChild>
                <Link href="/research">
                  연구 분야 보기
                </Link>
              </Button>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="relative border-t border-white/10 bg-white/5 backdrop-blur-sm">
          <div className="container py-8">
            <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
              {stats.map((stat) => (
                <div key={stat.label} className="text-center">
                  <div className="text-3xl font-bold text-white md:text-4xl">{stat.value}</div>
                  <div className="mt-1 text-sm text-slate-400">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Recent Blog Posts Section */}
      <section className="py-20 md:py-28">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <Badge variant="outline" className="mb-4">Blog</Badge>
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">최근 블로그 포스트</h2>
            <p className="mt-4 text-muted-foreground">
              최신 논문 리뷰와 AI 관련 글을 확인하세요
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
                더 많은 글 보기 <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* YouTube Section - Learn Through Video Tutorials */}
      <section className="bg-muted/30 py-20 md:py-28">
        <div className="container">
          <div className="grid gap-12 lg:grid-cols-2 lg:items-center">
            <div>
              <Badge className="mb-4">YouTube Channel</Badge>
              <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
                영상으로 배우기
              </h2>
              <p className="mt-4 text-muted-foreground">
                150개 이상의 교육용 비디오를 통해 데이터 과학, 머신러닝, 딥러닝, 파이썬 프로그래밍을 배워보세요.
              </p>
              <ul className="mt-8 space-y-3">
                {['Python Programming', 'Data Science', 'Machine Learning', 'Deep Learning', 'Computer Vision', 'NLP'].map((item) => (
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
                  영상 보기
                  <Youtube className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </div>
            <div className="aspect-video overflow-hidden rounded-xl shadow-2xl">
              <iframe
                className="h-full w-full"
                src="https://www.youtube.com/embed/k60oT_8lyFw"
                title="SuanLab YouTube"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
          </div>
        </div>
      </section>

      {/* Media Section - In the News */}
      <section className="py-20 md:py-28">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <Badge variant="outline" className="mb-4">
              <Newspaper className="mr-2 h-3 w-3" />
              미디어 보도
            </Badge>
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">미디어 보도</h2>
            <p className="mt-4 text-muted-foreground">
              SuanLab과 이수안 교수의 연구 및 활동에 관한 미디어 기사
            </p>
          </div>

          <div className="mt-12 grid gap-6 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {mediaArticles.map((article) => (
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
        </div>
      </section>

      {/* Quick Links - Explore SuanLab */}
      <section className="bg-muted/50 py-20 md:py-28">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">SuanLab 둘러보기</h2>
            <p className="mt-4 text-muted-foreground">
              논문, 프로젝트, 강의, 영상 튜토리얼을 살펴보세요
            </p>
          </div>

          <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {quickLinks.map((link) => (
              <Link key={link.href} href={link.href}>
                <Card className="group h-full transition-all hover:shadow-lg hover:border-primary/50">
                  <CardContent className="flex flex-col items-center p-6 text-center">
                    <div className="mb-4 rounded-full bg-primary/10 p-4 group-hover:bg-primary/20 transition-colors">
                      <link.icon className="h-8 w-8 text-primary" />
                    </div>
                    <h3 className="text-lg font-semibold">{link.title}</h3>
                    <p className="mt-2 text-sm text-muted-foreground">{link.description}</p>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Research Areas */}
      <section className="py-20 md:py-28">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">연구 분야</h2>
            <p className="mt-4 text-muted-foreground">
              데이터 사이언스와 인공지능 분야의 최신 기술을 연구합니다
            </p>
          </div>

          <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {researchAreas.map((area) => (
              <Link key={area.href} href={area.href}>
                <Card className="group h-full transition-all hover:shadow-lg hover:-translate-y-1">
                  <CardHeader>
                    <div className={`mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br ${area.color}`}>
                      <area.icon className="h-6 w-6 text-white" />
                    </div>
                    <CardTitle className="group-hover:text-primary transition-colors">
                      {area.title}
                    </CardTitle>
                    <CardDescription>{area.titleKo}</CardDescription>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gradient-to-r from-primary to-blue-600 py-20 text-white">
        <div className="container text-center">
          <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
            협업에 관심이 있으신가요?
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-white/80">
            데이터 사이언스와 인공지능 연구에 참여하거나 협업하고 싶으시면 연락 주세요.
          </p>
          <div className="mt-10 flex flex-col gap-4 sm:flex-row sm:justify-center">
            <Button size="lg" variant="secondary" asChild>
              <Link href="/suan">이수안 교수에게 연락하기</Link>
            </Button>
            <Button size="lg" variant="outline" className="bg-transparent border-white text-white hover:bg-white/10" asChild>
              <Link href="/publication">논문 보기</Link>
            </Button>
          </div>
        </div>
      </section>
    </>
  );
}
