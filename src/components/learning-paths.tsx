'use client';
import Link from 'next/link';
import { learningPaths } from '@/data/editorial';
import { useLanguage } from '@/components/language-provider';

export default function LearningPaths() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  return <section aria-labelledby="learning-paths" className="border-b bg-muted/20 py-12">
    <div className="container">
      <h2 id="learning-paths" className="text-2xl font-semibold">{ko ? '목표별 학습 경로' : 'Learning paths by goal'}</h2>
      <p className="mt-3 text-muted-foreground">{ko ? '선수 지식을 확인하고, 강의·영상·읽기 자료를 순서대로 따라가세요.' : 'Check the prerequisites, then follow courses, videos, and reading in order.'}</p>
      <div className="mt-6 grid gap-6 lg:grid-cols-3">{learningPaths.map(path => <article key={path.id} id={`path-${path.id}`} className="rounded-xl border bg-card p-6">
        <h3 className="text-lg font-semibold">{ko ? path.titleKo : path.titleEn}</h3>
        <p className="mt-2 text-sm text-muted-foreground">{ko ? `선수 지식: ${path.prerequisiteKo}` : `Prerequisites: ${path.prerequisiteEn}`}</p>
        <ol className="mt-6 space-y-4">{path.steps.map((step, i) => <li key={step.href} className="flex gap-3"><span aria-hidden="true" className="font-mono text-primary">0{i + 1}</span><div><p className="text-xs text-muted-foreground">{ko ? step.levelKo : step.levelEn}</p><Link className="text-sm hover:text-primary hover:underline" href={step.href}>{ko ? step.labelKo : step.labelEn}</Link></div></li>)}</ol>
      </article>)}</div>
    </div>
  </section>;
}
