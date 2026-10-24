'use client';
import { useLanguage } from '@/components/language-provider';

export function PublicationTimeline({ years, selected, onSelect }: {
  years: { year: number; count: number }[]; selected: string; onSelect: (year: string) => void;
}) {
  const { language } = useLanguage();
  const ko = language === 'ko';
  if (!years.length) return null;
  const maximum = Math.max(...years.map(item => item.count));
  return <section data-motion-reveal data-publication-timeline aria-labelledby="publication-timeline-title" className="mb-8 min-w-0 rounded-xl border bg-card p-5">
    <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
      <div><h2 id="publication-timeline-title" className="text-base font-semibold">{ko ? '연구 성과 타임라인' : 'Publication timeline'}</h2>
        <p className="mt-1 text-xs text-muted-foreground">{ko ? '전체 연구 성과의 연도별 분포 · 연도를 선택해 필터링하세요.' : 'All publications by year · Select a year to filter.'}</p></div>
      <button type="button" onClick={() => onSelect('all')} className="rounded-md border px-3 py-1.5 text-xs hover:bg-accent">{ko ? '전체 연도' : 'All years'}</button>
    </div>
    <div className="overflow-x-auto pb-2">
      <div className="flex min-w-max items-end gap-2" role="group" aria-label={ko ? '연도별 연구 성과' : 'Publications by year'}>
        {years.map(({ year, count }) => <button key={year} type="button" data-year={year} data-count={count} aria-pressed={selected === String(year)}
          aria-label={ko ? `${year}년 · ${count}편` : `${year} · ${count} publications`} onClick={() => onSelect(selected === String(year) ? 'all' : String(year))}
          className="group flex w-10 shrink-0 flex-col items-center gap-2 rounded-md px-1 py-2 hover:bg-accent aria-pressed:bg-primary/10">
          <span className="text-[10px] tabular-nums text-muted-foreground">{count}</span>
          <span aria-hidden="true" className="flex h-20 w-5 items-end"><span data-motion-bar className="block w-full rounded-t-sm bg-primary/40 transition-colors group-hover:bg-primary group-aria-pressed:bg-primary" style={{ height: `${Math.max(8, count / maximum * 80)}px` }} /></span>
          <span className="font-mono text-[10px] text-muted-foreground group-aria-pressed:text-primary">{year}</span>
        </button>)}
      </div>
    </div>
  </section>;
}
