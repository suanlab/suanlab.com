'use client';

import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
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

function particlesFor(pattern: DotVariant) {
  const motifCount = pattern === 'qt' ? 60 : 96;
  const ambientCount = pattern === 'qt' ? 72 : 96;
  return Array.from({ length: motifCount + ambientCount }, (_, i) => {
    const ambient = i >= motifCount;
    const n = ambient ? i - motifCount : i;
    const p = point(pattern, n);
    // A stratified particle layer fills every edge, independent of the menu motif.
    const x = ambient ? ((n % 12 + .2 + (n * 17 % 7) / 10) / 12) * 100 : 3 + (p.x - 100) / 5.5;
    const y = ambient ? ((Math.floor(n / 12) + .2 + (n * 13 % 7) / 10) / (ambientCount / 12)) * 100 : 3 + p.y / 2.55;
    return { i, n, ambient, x: Number(Math.min(97, Math.max(3, x)).toFixed(2)), y: Number(Math.min(97, Math.max(3, y)).toFixed(2)) };
  });
}

function connectionsFor(particles: ReturnType<typeof particlesFor>) {
  // Sample a few visible dots; cap degree and edge count to keep the graph sparse.
  const nodes = particles.filter(p => p.n % (p.ambient ? 10 : 4) === 0);
  const edges: { from: number; to: number }[] = [];
  const degree = new Map<number, number>();
  const seen = new Set<string>();
  for (const a of nodes) {
    const neighbors = nodes.filter(b => b.i !== a.i).map(b => ({ b, distance: Math.hypot((a.x - b.x) * 3, a.y - b.y) })).sort((a, b) => a.distance - b.distance);
    for (const { b, distance } of neighbors) {
      const key = [a.i, b.i].sort((a, b) => a - b).join('-');
      if (distance < 3 || distance > 80 || seen.has(key) || (degree.get(a.i) || 0) >= 3 || (degree.get(b.i) || 0) >= 3) continue;
      edges.push({ from: a.i, to: b.i });
      seen.add(key);
      degree.set(a.i, (degree.get(a.i) || 0) + 1);
      degree.set(b.i, (degree.get(b.i) || 0) + 1);
      break;
    }
    if (edges.length === 32) break;
  }
  return edges;
}

export default function MenuDotField({ variant }: { variant?: DotVariant }) {
  const pathname = usePathname();
  const pattern = variant || menus[pathname?.split('/')[1] || ''] || 'research';
  const { language } = useLanguage();
  const ref = useRef<HTMLDivElement>(null);
  const [enabled, setEnabled] = useState(false);
  const [reduced, setReduced] = useState(true);
  const [paused, setPaused] = useState(false);
  const connectionRef = useRef<SVGSVGElement>(null);
  const particles = useMemo(() => particlesFor(pattern), [pattern]);
  const connections = useMemo(() => connectionsFor(particles), [particles]);
  const running = enabled && !paused;

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

  useEffect(() => {
    const field = ref.current;
    const svg = connectionRef.current;
    if (!field || !svg) return;
    const ids = new Set(connections.flatMap(edge => [edge.from, edge.to]));
    const nodes = [...field.querySelectorAll<HTMLElement>('.menu-dot-anchor')]
      .filter(node => ids.has(Number(node.dataset.node)))
      .map(node => ({ id: Number(node.dataset.node), dot: node.firstElementChild! }));
    const lines = [...svg.querySelectorAll('line')];
    // Read all dot positions before writing SVG coordinates; no React updates per frame.
    const sync = () => {
      const bounds = field.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      const positions = new Map(nodes.map(({ id, dot }) => {
        const rect = dot.getBoundingClientRect();
        return [id, { x: rect.left + rect.width / 2 - bounds.left, y: rect.top + rect.height / 2 - bounds.top }] as const;
      }));
      const reach = Math.min(320, Math.max(150, bounds.width * .25));
      connections.forEach((edge, i) => {
        const a = positions.get(edge.from);
        const b = positions.get(edge.to);
        if (!a || !b) return;
        const line = lines[i];
        line.setAttribute('x1', (a.x / bounds.width * 100).toFixed(3));
        line.setAttribute('y1', (a.y / bounds.height * 100).toFixed(3));
        line.setAttribute('x2', (b.x / bounds.width * 100).toFixed(3));
        line.setAttribute('y2', (b.y / bounds.height * 100).toFixed(3));
        line.setAttribute('stroke-opacity', Math.max(0, 1 - Math.hypot(a.x - b.x, a.y - b.y) / reach).toFixed(3));
      });
    };
    let frame = 0;
    let last = 0;
    const tick = (time: number) => {
      if (time - last >= 1000 / 30) { sync(); last = time; }
      frame = requestAnimationFrame(tick);
    };
    sync();
    const observer = new ResizeObserver(sync);
    observer.observe(field);
    if (running) frame = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(frame); observer.disconnect(); };
  }, [connections, running, reduced]);

  const palette = [colors[pattern], '#38bdf8', '#a78bfa', '#f472b6', '#fbbf24', '#34d399'];

  return <>
    <div ref={ref} aria-hidden="true" className="menu-dot-field" data-pattern={pattern} data-motion={running ? 'running' : 'paused'}>
      <svg ref={connectionRef} className="menu-connections" viewBox="0 0 100 100" preserveAspectRatio="none" focusable="false">
        {connections.map((edge, i) => <line key={`${edge.from}-${edge.to}`} className="menu-connection" data-from={edge.from} data-to={edge.to}
          x1={particles[edge.from].x} y1={particles[edge.from].y} x2={particles[edge.to].x} y2={particles[edge.to].y}
          stroke={palette[i % palette.length]} vectorEffect="non-scaling-stroke" strokeDasharray={i % 5 === 0 ? '3 9' : undefined} style={{ animationDelay: `${-i * .3}s` }} />)}
      </svg>
      {particles.map(({ i, n, ambient, x, y }) => {
        const style = {
          left: `${x}%`,
          top: `${y}%`,
          '--dot-color': palette[i % palette.length],
          '--dot-size': `${n % 9 === 0 ? 5 : n % 3 === 0 ? 3.5 : 2.5}px`,
          '--dx': `${((pattern === 'contact' ? (n % 2 ? -1 : 1) : Math.cos(n * 2.4)) * (ambient ? 65 : 48)).toFixed(2)}px`,
          '--dy': `${(Math.sin(n * 2.4) * (ambient ? 48 : 32)).toFixed(2)}px`,
          '--delay': `${pattern === 'deadlines' && !ambient ? -n / 8 : -n * .21}s`,
          '--duration': `${(ambient ? 9 : 4) + n % 7}s`,
        } as CSSProperties;
        return <span key={i} data-node={i} className={`menu-dot-anchor ${ambient ? 'menu-dot-ambient' : 'menu-dot-motif'}`} style={style}>
          <span className={`menu-dot ${ambient ? 'menu-dot-drift' : `menu-dot-${pattern}`}`} />
        </span>;
      })}
    </div>
    {!reduced && <button type="button" className="menu-dot-control absolute right-4 top-3 z-20 inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/20 bg-slate-950/65 text-cyan-100 hover:bg-slate-800" onClick={() => setPaused(current => !current)} aria-label={language === 'ko' ? (paused ? '제목 배경 애니메이션 재생' : '제목 배경 애니메이션 일시정지') : (paused ? 'Play title background animation' : 'Pause title background animation')}>
      {paused ? <Play aria-hidden="true" className="h-3.5 w-3.5" /> : <Pause aria-hidden="true" className="h-3.5 w-3.5" />}
    </button>}
  </>;
}
