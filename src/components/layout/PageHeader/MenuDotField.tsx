'use client';

import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { usePathname } from 'next/navigation';
import { Pause, Play } from 'lucide-react';
import { useLanguage } from '@/components/language-provider';
import './menu-dots.css';

type DotVariant = 'profile' | 'research' | 'project' | 'publication' | 'blog' | 'book' | 'lecture' | 'course' | 'youtube' | 'prompts' | 'deadlines' | 'contact' | 'search' | 'privacy' | 'terms' | 'qt';
const menus: Record<string, DotVariant> = {
  suan: 'profile', research: 'research', project: 'project', publication: 'publication',
  blog: 'blog', book: 'book', lecture: 'lecture', course: 'course', youtube: 'youtube',
  prompts: 'prompts', deadlines: 'deadlines', contact: 'contact', search: 'search',
  privacy: 'privacy', terms: 'terms', qt: 'qt',
};
const colors: Record<DotVariant, string> = {
  profile: '#a5b4fc', research: '#67e8f9', project: '#5eead4', publication: '#93c5fd',
  blog: '#c4b5fd', book: '#fcd34d', lecture: '#7dd3fc', course: '#a7f3d0',
  youtube: '#fda4af', prompts: '#c4b5fd', deadlines: '#fdba74', contact: '#5eead4',
  search: '#67e8f9', privacy: '#a5b4fc', terms: '#cbd5e1', qt: '#fcd34d',
};

// Deterministic positions keep the static HTML and hydrated client identical.
function point(variant: DotVariant, i: number) {
  const col = i % 12;
  const row = Math.floor(i / 12);
  const angle = (i % 24) * Math.PI / 12;
  switch (variant) {
    case 'profile': {
      const radius = 28 + Math.floor(i / 24) * 24;
      return { x: 350 + Math.cos(angle) * radius * 1.4, y: 120 + Math.sin(angle) * radius * .8 };
    }
    case 'research': {
      const cluster = i % 3;
      const radius = 10 + (Math.floor(i / 3) % 8) * 5;
      return { x: 195 + cluster * 155 + Math.cos(i * 2.4) * radius, y: 110 + Math.sin(i * 2.4) * radius };
    }
    case 'project': return { x: 160 + (i % 4) * 115 + (Math.floor(i / 32) % 3) * 9, y: 65 + (Math.floor(i / 4) % 8) * 15 };
    case 'publication': return { x: 195 + col * 27, y: 55 + row * 18 };
    case 'blog': return { x: 165 + col * 27 + (row % 2) * 12, y: 55 + row * 19 };
    case 'book': return { x: 155 + (i % 2) * 220 + (Math.floor(i / 2) % 6) * 24, y: 62 + Math.floor(i / 12) * 18 - Math.sin((Math.floor(i / 2) % 6) * Math.PI / 7) * 12 };
    case 'lecture': return { x: 160 + col * 29, y: 198 - row * 14 - col * 4 };
    case 'course': {
      const radius = 24 + Math.floor(i / 24) * 24;
      return { x: 350 + Math.cos(angle) * radius, y: 120 + Math.sin(angle) * radius };
    }
    case 'youtube': return { x: 175 + col * 28, y: 195 - row * (9 + Math.abs(Math.sin(col * .8)) * 12) };
    case 'prompts': return { x: 180 + col * 28, y: 55 + row * 19 };
    case 'deadlines': return { x: 350 + Math.sin(i * Math.PI / 48) * 85, y: 120 - Math.cos(i * Math.PI / 48) * 85 };
    case 'contact': return { x: (i % 2 ? 425 : 115) + (Math.floor(i / 2) % 12) * 10, y: 74 + Math.floor(i / 24) * 28 };
    case 'search': {
      if (i >= 72) return { x: 397 + (i - 72) * 2.3, y: 153 + (i - 72) * 2.3 };
      return { x: 350 + Math.cos(i * Math.PI / 36) * 63, y: 105 + Math.sin(i * Math.PI / 36) * 63 };
    }
    case 'privacy': {
      const corners = [[280, 50], [350, 30], [420, 50], [413, 133], [350, 210], [287, 133], [280, 50]];
      const segment = Math.floor(i / 16);
      const step = (i % 16) / 16;
      return { x: corners[segment][0] + (corners[segment + 1][0] - corners[segment][0]) * step, y: corners[segment][1] + (corners[segment + 1][1] - corners[segment][1]) * step };
    }
    case 'terms': return { x: 195 + col * 27, y: 55 + row * 19 + (col < 2 ? col * 5 : 0) };
    case 'qt': return { x: 120 + (i * 71) % 430, y: 35 + (i * 53) % 170 };
  }
}

export default function MenuDotField({ variant }: { variant?: DotVariant }) {
  const pathname = usePathname();
  const pattern = variant || menus[pathname?.split('/')[1] || ''] || 'research';
  const { language } = useLanguage();
  const ref = useRef<HTMLDivElement>(null);
  const [enabled, setEnabled] = useState(false);
  const [reduced, setReduced] = useState(true);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    let inView = false;
    const update = () => { setReduced(preference.matches); setEnabled(inView && !document.hidden && !preference.matches); };
    const observer = new IntersectionObserver(([entry]) => { inView = entry.isIntersecting; update(); });
    if (ref.current) observer.observe(ref.current);
    update();
    preference.addEventListener('change', update);
    document.addEventListener('visibilitychange', update);
    return () => { observer.disconnect(); preference.removeEventListener('change', update); document.removeEventListener('visibilitychange', update); };
  }, []);

  return <>
    <div ref={ref} aria-hidden="true" className="menu-dot-field" data-pattern={pattern} data-motion={enabled && !paused ? 'running' : 'paused'} style={{ '--dot-color': colors[pattern] } as CSSProperties}>
      <svg viewBox="0 0 700 240" preserveAspectRatio="xMidYMid meet" className="h-full w-full">
        {Array.from({ length: pattern === 'qt' ? 60 : 96 }, (_, i) => {
          const p = point(pattern, i);
          const style = {
            '--dx': `${pattern === 'contact' ? (i % 2 ? -90 : 90) : (Math.cos(i * 2.4) * 24).toFixed(2)}px`,
            '--dy': `${(Math.sin(i * 2.4) * 16).toFixed(2)}px`,
            '--delay': `${pattern === 'deadlines' ? -i / 8 : -i * .14}s`,
          } as CSSProperties;
          return <circle key={i} cx={p.x.toFixed(2)} cy={p.y.toFixed(2)} r={i % 9 === 0 ? 3.1 : 2.1} className={`menu-dot menu-dot-${pattern}`} style={style} />;
        })}
      </svg>
    </div>
    {!reduced && <button type="button" className="menu-dot-control absolute right-4 top-3 z-20 inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/20 bg-slate-950/65 text-cyan-100 hover:bg-slate-800" onClick={() => setPaused(current => !current)} aria-label={language === 'ko' ? (paused ? '제목 배경 애니메이션 재생' : '제목 배경 애니메이션 일시정지') : (paused ? 'Play title background animation' : 'Pause title background animation')}>
      {paused ? <Play aria-hidden="true" className="h-3.5 w-3.5" /> : <Pause aria-hidden="true" className="h-3.5 w-3.5" />}
    </button>}
  </>;
}
