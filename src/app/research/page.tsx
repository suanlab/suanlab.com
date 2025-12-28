import Image from 'next/image';
import Link from 'next/link';
import { Database, Brain, BookOpen, Eye, Network, MapPin, Mic } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { researchAreas } from '@/data/research';

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

export const metadata = {
  title: 'Research | SuanLab',
  description: 'SuanLab Research Areas - Data Science, Deep Learning, NLP, Computer Vision',
};

export default function ResearchPage() {
  return (
    <>
      <PageHeader
        title="Research"
        subtitle="Exploring cutting-edge technologies in data science and artificial intelligence"
        breadcrumbs={[{ label: 'Research' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-12">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
              Research <span className="text-primary">Areas</span>
            </h2>
            <p className="mt-4 text-muted-foreground">
              SuanLab focuses on various research areas in data science and AI
            </p>
          </div>

          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {researchAreas.map((area) => {
              const Icon = iconMap[area.icon] || Database;
              const color = colorMap[area.slug] || 'from-blue-500 to-cyan-500';

              return (
                <Link key={area.slug} href={`/research/${area.slug}`}>
                  <Card className="group h-full overflow-hidden transition-all hover:shadow-lg hover:-translate-y-1">
                    <div className="relative aspect-video overflow-hidden">
                      <Image
                        src={area.image}
                        alt={area.titleEn}
                        fill
                        className="object-cover transition-transform group-hover:scale-105"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                      <div className={`absolute top-4 left-4 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br ${color}`}>
                        <Icon className="h-5 w-5 text-white" />
                      </div>
                    </div>
                    <CardContent className="p-6">
                      <h3 className="text-lg font-semibold group-hover:text-primary transition-colors">
                        {area.titleEn}
                      </h3>
                      <p className="mt-1 text-sm text-muted-foreground">
                        {area.titleKo}
                      </p>
                      <p className="mt-3 text-sm text-muted-foreground line-clamp-2">
                        {area.description}
                      </p>
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
