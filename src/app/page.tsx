import Link from 'next/link';
import { ArrowRight, Brain, Database, Eye, BarChart3, Network, MapPin, Youtube, BookOpen, Newspaper, FolderKanban, AudioLines, ExternalLink, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { mediaArticles } from '@/data/media';

const researchAreas = [
  { title: 'Data Science & Big Data', titleKo: '데이터과학 및 빅데이터', icon: Database, href: '/research/ds', color: 'from-blue-500 to-cyan-500' },
  { title: 'Deep Learning & ML', titleKo: '딥러닝 및 머신러닝', icon: Brain, href: '/research/dl', color: 'from-purple-500 to-pink-500' },
  { title: 'Natural Language Processing', titleKo: '자연어처리', icon: BookOpen, href: '/research/nlp', color: 'from-green-500 to-emerald-500' },
  { title: 'Computer Vision', titleKo: '컴퓨터 비전', icon: Eye, href: '/research/cv', color: 'from-orange-500 to-red-500' },
  { title: 'Graphs and Tensors', titleKo: '그래프 및 텐서', icon: Network, href: '/research/graphs', color: 'from-indigo-500 to-violet-500' },
  { title: 'Spatio-Temporal', titleKo: '시공간 데이터', icon: MapPin, href: '/research/st', color: 'from-teal-500 to-cyan-500' },
  { title: 'Audio & Speech Processing', titleKo: '오디오 음성 처리', icon: AudioLines, href: '/research/asp', color: 'from-rose-500 to-pink-500' },
];

const stats = [
  { label: 'Publications', value: '209+' },
  { label: 'YouTube Videos', value: '167+' },
  { label: 'Projects', value: '46' },
  { label: 'Lectures', value: '16' },
];

const quickLinks = [
  { title: 'Research', description: 'Explore our research areas', icon: BarChart3, href: '/research' },
  { title: 'YouTube', description: 'Watch tutorial videos', icon: Youtube, href: '/youtube' },
  { title: 'Publications', description: 'Browse our papers', icon: Newspaper, href: '/publication' },
  { title: 'Projects', description: 'View our projects', icon: FolderKanban, href: '/project' },
];

export default function Home() {
  return (
    <>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
        <div className="absolute inset-0 bg-[url('/assets/images/slider/2.jpg')] bg-cover bg-center opacity-20" />
        <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-transparent" />

        <div className="container relative py-24 md:py-32 lg:py-40">
          <div className="mx-auto max-w-3xl text-center">
            <Badge variant="secondary" className="mb-4">
              Data Science & Artificial Intelligence
            </Badge>
            <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
              Welcome to{' '}
              <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                SuanLab
              </span>
            </h1>
            <p className="mt-6 text-lg text-slate-300 md:text-xl">
              Research lab focused on Data Science, Deep Learning, Machine Learning, and Big Data.
              Sharing knowledge through publications, lectures, and YouTube content.
            </p>
            <div className="mt-10 flex flex-col gap-4 sm:flex-row sm:justify-center">
              <Button size="lg" asChild>
                <Link href="/suan">
                  About Professor Suan
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" className="bg-white/10 hover:bg-white/20" asChild>
                <Link href="/research">
                  Explore Research
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

      {/* Research Areas */}
      <section className="py-20 md:py-28">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">Research Areas</h2>
            <p className="mt-4 text-muted-foreground">
              Exploring cutting-edge technologies in data science and artificial intelligence
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

      {/* Quick Links */}
      <section className="bg-muted/50 py-20 md:py-28">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">Explore SuanLab</h2>
            <p className="mt-4 text-muted-foreground">
              Discover publications, projects, lectures, and video tutorials
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

      {/* Media Section */}
      <section className="py-20 md:py-28">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <Badge variant="outline" className="mb-4">
              <Newspaper className="mr-2 h-3 w-3" />
              Media Coverage
            </Badge>
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">In the News</h2>
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

      {/* YouTube Section */}
      <section className="bg-muted/30 py-20 md:py-28">
        <div className="container">
          <div className="grid gap-12 lg:grid-cols-2 lg:items-center">
            <div>
              <Badge className="mb-4">YouTube Channel</Badge>
              <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
                Learn Through Video Tutorials
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
                  Watch Videos
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

      {/* CTA Section */}
      <section className="bg-gradient-to-r from-primary to-blue-600 py-20 text-white">
        <div className="container text-center">
          <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
            Interested in Collaboration?
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-white/80">
            Contact us if you want to join or collaborate on data science and artificial intelligence research.
          </p>
          <div className="mt-10 flex flex-col gap-4 sm:flex-row sm:justify-center">
            <Button size="lg" variant="secondary" asChild>
              <Link href="/suan">Contact Professor Suan</Link>
            </Button>
            <Button size="lg" variant="outline" className="bg-transparent border-white text-white hover:bg-white/10" asChild>
              <Link href="/publication">View Publications</Link>
            </Button>
          </div>
        </div>
      </section>
    </>
  );
}
