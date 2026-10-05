'use client';
import Link from 'next/link';
import { researchLinks } from '@/data/editorial';
import { researchAreas } from '@/data/research';
import { useLanguage } from '@/components/language-provider';

export default function ResearchBacklinks({ projectId, lectureSlug }: { projectId?: number; lectureSlug?: string }) {
  const { language } = useLanguage();
  const areas = researchAreas.filter(area => {
    const links = researchLinks[area.slug];
    return links && ((projectId !== undefined && links.projects.includes(projectId)) || (lectureSlug && links.lectures.includes(lectureSlug)));
  });
  if (!areas.length) return null;
  return <nav aria-label={language === 'ko' ? '관련 연구 분야' : 'Related research areas'} className="mt-6 border-t pt-4">
    <p className="mb-3 text-xs font-medium text-muted-foreground">{language === 'ko' ? '주제로 연결된 연구 분야' : 'Research areas sharing this topic'}</p>
    <div className="flex flex-wrap gap-2">{areas.map(area => <Link key={area.slug} href={`/research/${area.slug}/`} className="rounded-full border px-3 py-1 text-xs hover:border-primary hover:text-primary">{language === 'ko' ? area.titleKo : area.titleEn}</Link>)}</div>
  </nav>;
}
