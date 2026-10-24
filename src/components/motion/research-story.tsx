'use client';

import { useEffect, useId, useRef, useState } from 'react';
import Link from 'next/link';
import { ArrowUpRight, Database, Network, ScanLine, Sparkles } from 'lucide-react';
import { useLanguage } from '@/components/language-provider';
import { useSceneActivity } from './use-scene-activity';

const stages = [
  { id: 'data', label: 'DATA', icon: Database, x: 95, y: 90, title: ['데이터에서 질문을 찾고', 'Start with data and questions'], description: ['텍스트·이미지·음성·시공간 데이터의 구조와 맥락을 살펴봅니다.', 'Explore the structure and context of text, images, audio and spatiotemporal data.'], detail: ['입력 · 데이터 품질 · 문제 정의', 'Inputs · Data quality · Problem definition'] },
  { id: 'models', label: 'MODELS', icon: Network, x: 325, y: 90, title: ['모델과 지식을 연결하고', 'Connect models and knowledge'], description: ['신경망, 언어 모델, 검색 증강 생성과 에이전트로 문제 해결 방법을 설계합니다.', 'Design approaches with neural networks, language models, retrieval and agents.'], detail: ['표현 · 검색 · 추론', 'Representations · Retrieval · Reasoning'] },
  { id: 'evidence', label: 'EVIDENCE', icon: ScanLine, x: 325, y: 280, title: ['실험과 근거로 검증하고', 'Evaluate against evidence'], description: ['벤치마크와 실험을 통해 성능, 한계와 재현 가능성을 확인합니다.', 'Use benchmarks and experiments to examine performance, limitations and reproducibility.'], detail: ['실험 · 평가 · 재현성', 'Experiments · Evaluation · Reproducibility'] },
  { id: 'impact', label: 'IMPACT', icon: Sparkles, x: 95, y: 280, title: ['실제 가치와 배움으로', 'Bring research into practice'], description: ['논문, 응용 프로젝트와 공개 교육 자료로 연구 결과를 나눕니다.', 'Share research through papers, applied projects and open educational resources.'], detail: ['논문 · 시스템 · 교육', 'Papers · Systems · Education'] },
];

function ProcessGraphic({ active }: { active: number }) {
  const gradient = `process-${useId().replace(/:/g, '')}`;
  return <svg aria-hidden="true" focusable="false" viewBox="0 0 420 360" className="mx-auto w-full max-w-md" fill="none">
    <defs><linearGradient id={gradient}><stop stopColor="#0891b2" /><stop offset="1" stopColor="#8b5cf6" /></linearGradient></defs>
    {[74, 112, 150].map(radius => <circle key={radius} cx="210" cy="180" r={radius} stroke="currentColor" className="text-border" strokeDasharray={radius === 150 ? '2 10' : undefined} />)}
    {stages.map((stage, index) => {
      const next = stages[(index + 1) % stages.length];
      const d = `M${stage.x} ${stage.y} Q210 180 ${next.x} ${next.y}`;
      return <g key={stage.id}>
        <path d={d} stroke="currentColor" className="text-border" strokeWidth="1.5" />
        <path d={d} className="story-edge" pathLength="1" stroke={`url(#${gradient})`} strokeWidth="2" strokeDasharray="1" strokeDashoffset={index < active ? 0 : 1} />
        {index === active && <path d={d} className="story-signal" pathLength="100" stroke={`url(#${gradient})`} strokeWidth="3" strokeLinecap="round" strokeDasharray="5 95" />}
      </g>;
    })}
    <circle cx="210" cy="180" r="45" className="fill-background" stroke={`url(#${gradient})`} />
    <text x="210" y="177" textAnchor="middle" fill="currentColor" className="text-foreground" fontSize="13" fontWeight="600">SUANLAB</text>
    <text x="210" y="194" textAnchor="middle" fill="currentColor" className="text-muted-foreground" fontSize="8" letterSpacing="2">RESEARCH FLOW</text>
    {stages.map((stage, index) => <g key={stage.id} className="story-node" opacity={active === index ? 1 : .65}>
      <circle cx={stage.x} cy={stage.y} r={active === index ? 30 : 25} className="fill-background" stroke={active === index ? `url(#${gradient})` : 'currentColor'} strokeWidth={active === index ? 2 : 1} />
      <text x={stage.x} y={stage.y + 4} textAnchor="middle" fill="currentColor" fontSize="12" className="text-primary" fontWeight="600">0{index + 1}</text>
      <text x={stage.x} y={stage.y + 48} textAnchor="middle" fill="currentColor" className="text-foreground" fontSize="10" letterSpacing="1.5">{stage.label}</text>
    </g>)}
  </svg>;
}

export function ResearchStory({ compact = false }: { compact?: boolean }) {
  const { language } = useLanguage();
  const lang = language === 'ko' ? 0 : 1;
  const [active, setActive] = useState(0);
  const ref = useRef<HTMLElement>(null);
  const running = useSceneActivity(ref);
  useEffect(() => {
    if (compact || !ref.current || !('IntersectionObserver' in window)) return;
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) setActive(Number((visible.target as HTMLElement).dataset.storyStep));
    }, { rootMargin: '-20% 0px -35% 0px', threshold: [0, .25, .5, .75] });
    ref.current.querySelectorAll('[data-story-step]').forEach(step => observer.observe(step));
    return () => observer.disconnect();
  }, [compact]);
  const prefix = compact ? 'home-process' : 'research-process';
  return <section ref={ref} data-motion={running ? 'running' : 'paused'} data-research-story={compact ? 'compact' : 'full'} aria-labelledby={`${prefix}-title`} className="research-story border-b py-16 md:py-20">
    <div className="container">
      <div data-motion-reveal className="mb-10 max-w-2xl">
        <p className="mb-3 font-mono text-xs uppercase tracking-[.22em] text-primary">From data to intelligence</p>
        <h2 id={`${prefix}-title`} className="text-3xl font-semibold tracking-tight md:text-4xl">{lang === 0 ? '데이터에서 시작해, 실제 가치로.' : 'From data to real-world value.'}</h2>
        <p className="mt-4 leading-7 text-muted-foreground">{lang === 0 ? '질문을 세우고, 연결하고, 검증하고, 나누는 연구의 흐름입니다.' : 'A research process of asking, connecting, evaluating and sharing.'}</p>
      </div>
      <div className={compact ? 'grid items-center gap-8 lg:grid-cols-[.8fr_1.2fr]' : 'grid items-start gap-10 lg:grid-cols-2 lg:gap-20'}>
        <div className={compact ? 'hidden lg:block' : 'lg:sticky lg:top-24'}>
          <ProcessGraphic active={active} />
          {!compact && <nav aria-label={lang === 0 ? '연구 단계' : 'Research stages'} className="mt-2 flex flex-wrap justify-center gap-2">
            {stages.map((stage, index) => <a key={stage.id} href={`#${prefix}-${stage.id}`} onClick={() => setActive(index)} aria-current={active === index ? 'step' : undefined}
              className="rounded-full border px-3 py-2 font-mono text-[10px] tracking-wider transition-colors hover:border-primary hover:text-primary aria-[current=step]:border-primary aria-[current=step]:text-primary">0{index + 1} {stage.label}</a>)}
          </nav>}
        </div>
        <div className={compact ? 'grid gap-4 sm:grid-cols-2' : 'space-y-6'}>
          {stages.map((stage, index) => <article key={stage.id} id={`${prefix}-${stage.id}`} data-story-step={index} data-active={active === index} data-motion-reveal
            onMouseEnter={compact ? () => setActive(index) : undefined} className={`story-step rounded-2xl border bg-card p-6 ${compact ? '' : 'flex min-h-64 flex-col justify-center md:min-h-80 md:p-8'}`}>
            <div className="mb-5 flex items-center justify-between text-primary"><stage.icon aria-hidden="true" className="h-5 w-5" /><span className="font-mono text-xs tracking-wider">0{index + 1} / {stage.label}</span></div>
            <h3 className="text-lg font-semibold md:text-xl">{stage.title[lang]}</h3>
            <p className="mt-3 text-sm leading-7 text-muted-foreground">{stage.description[lang]}</p>
            {!compact && <p className="mt-6 border-t pt-4 text-xs text-primary">{stage.detail[lang]}</p>}
          </article>)}
        </div>
      </div>
      <Link href={compact ? '/research/' : '/publication/'} className="mt-8 inline-flex items-center gap-2 text-sm font-medium text-primary hover:underline">
        {compact ? (lang === 0 ? '연구의 흐름 살펴보기' : 'Explore the research process') : (lang === 0 ? '연구 성과 살펴보기' : 'Explore our publications')}<ArrowUpRight aria-hidden="true" className="h-4 w-4" />
      </Link>
    </div>
  </section>;
}
