import Image from 'next/image';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { Database, Brain, BookOpen, Eye, Network, MapPin, ArrowRight } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { researchAreas, getResearchBySlug } from '@/data/research';

const iconMap: Record<string, React.ElementType> = {
  'et-gears': Database,
  'et-layers': Brain,
  'et-document': BookOpen,
  'et-pictures': Eye,
  'et-linegraph': Network,
  'et-map': MapPin,
};

const colorMap: Record<string, string> = {
  ds: 'from-blue-500 to-cyan-500',
  dl: 'from-purple-500 to-pink-500',
  nlp: 'from-green-500 to-emerald-500',
  cv: 'from-orange-500 to-red-500',
  graphs: 'from-indigo-500 to-violet-500',
  st: 'from-teal-500 to-cyan-500',
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

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="grid gap-12 lg:grid-cols-2 lg:items-center">
            <div className="relative aspect-video overflow-hidden rounded-xl shadow-2xl">
              <Image
                src={area.image}
                alt={area.titleEn}
                fill
                className="object-cover"
              />
            </div>
            <div>
              <div className={`mb-6 inline-flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br ${color}`}>
                <Icon className="h-7 w-7 text-white" />
              </div>
              <h2 className="text-2xl font-bold md:text-3xl">{area.titleEn}</h2>
              <p className="mt-2 text-lg text-muted-foreground">{area.titleKo}</p>
              <p className="mt-6 text-muted-foreground leading-relaxed">
                {area.description}
              </p>
              <p className="mt-4 text-muted-foreground leading-relaxed">
                SuanLab에서는 {area.titleKo} 분야에 대한 연구를 진행하고 있습니다.
                다양한 프로젝트와 논문을 통해 이 분야의 발전에 기여하고 있습니다.
              </p>
              <div className="mt-8 flex flex-wrap gap-4">
                <Button asChild>
                  <Link href="/publication">
                    View Publications
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button variant="outline" asChild>
                  <Link href="/project">
                    View Projects
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-muted/50 py-16 md:py-20">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-12">
            <h2 className="text-2xl font-bold md:text-3xl">
              Related <span className="text-primary">Publications</span>
            </h2>
            <p className="mt-4 text-muted-foreground">
              이 분야와 관련된 출판물은 Publication 페이지에서 확인하실 수 있습니다.
            </p>
          </div>
          <div className="text-center">
            <Button size="lg" asChild>
              <Link href="/publication">
                Browse All Publications
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

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
                    <Card className="group transition-all hover:shadow-md hover:border-primary/50">
                      <CardContent className="flex items-center gap-4 p-4">
                        <div className={`flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br ${rColor}`}>
                          <RIcon className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <h3 className="font-semibold group-hover:text-primary transition-colors">
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
