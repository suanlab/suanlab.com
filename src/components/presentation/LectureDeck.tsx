'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import 'katex/dist/katex.min.css';
import { slideIndex, type Slide } from '@/lib/presentation';

export default function LectureDeck({ slides, slug, title }: { slides: Slide[]; slug: string; title: string }) {
  const [index, setIndex] = useState(0);
  const [notes, setNotes] = useState(false);
  const [message, setMessage] = useState('');
  const root = useRef<HTMLDivElement>(null);
  const move = useCallback((value: number) => {
    const next = Math.max(0, Math.min(slides.length - 1, value));
    setIndex(next);
    window.history.replaceState(null, '', `#${next + 1}`);
  }, [slides.length]);
  useEffect(() => {
    document.body.dataset.presentation = 'true';
    const sync = () => setIndex(slideIndex(window.location.hash, slides.length));
    sync(); window.addEventListener('hashchange', sync);
    return () => { delete document.body.dataset.presentation; window.removeEventListener('hashchange', sync); };
  }, [slides.length]);
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || (event.target instanceof HTMLElement && event.target.closest('input, textarea, select, [contenteditable]'))) return;
      if (event.key === ' ' && event.target instanceof HTMLElement && event.target.closest('button, a')) return;
      if (['ArrowRight', 'ArrowDown', 'PageDown', ' '].includes(event.key)) { event.preventDefault(); move(index + 1); }
      if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(event.key)) { event.preventDefault(); move(index - 1); }
      if (event.key === 'Home') { event.preventDefault(); move(0); }
      if (event.key === 'End') { event.preventDefault(); move(slides.length - 1); }
    };
    window.addEventListener('keydown', key); return () => window.removeEventListener('keydown', key);
  }, [index, move, slides.length]);
  async function fullscreen() {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await root.current?.requestFullscreen(); }
    catch { setMessage('이 브라우저에서는 전체 화면을 사용할 수 없습니다.'); }
  }
  return <div ref={root} className="lecture-deck overflow-y-auto min-h-screen bg-background text-foreground">
    <div className="deck-controls sticky top-0 z-10 flex flex-wrap items-center justify-between gap-3 border-b bg-background p-4">
      <Link href={`/lecture/${slug}/`} className="text-sm text-primary">← {title}</Link>
      <div className="flex flex-wrap gap-2"><Button variant="outline" size="sm" onClick={() => setNotes(!notes)} aria-pressed={notes}>발표 노트</Button><Button variant="outline" size="sm" onClick={fullscreen}>전체 화면</Button><Button variant="outline" size="sm" onClick={() => window.print()}>인쇄 / PDF</Button></div>
    </div>
    <p className="sr-only" role="status">{message || `${index + 1} / ${slides.length}: ${slides[index].title}`}</p>
    {slides.map((slide, i) => <section key={i} aria-hidden={i !== index} className={`deck-slide mx-auto flex min-h-[70vh] max-w-5xl flex-col justify-center px-6 py-12 md:px-12 ${i === index ? '' : 'hidden'}`}>
      <p className="mb-6 font-mono text-xs uppercase tracking-widest text-primary">SuanLab / {slides.some(s => s.contentHtml) ? 'Lecture' : 'Lecture overview'} / {String(i + 1).padStart(2, '0')}</p>
      <h1 className="mb-10 [word-break:keep-all] text-3xl font-semibold leading-tight md:text-5xl">{slide.title}</h1>
      {slide.contentHtml ? <div className="lecture-prose" dangerouslySetInnerHTML={{ __html: slide.contentHtml }} /> : <ul className="space-y-6 text-lg leading-relaxed md:text-2xl">{slide.points.map((point) => <li key={point} className="flex gap-4"><span aria-hidden="true" className="text-primary">—</span><span>{point}</span></li>)}</ul>}
      {notes && slide.notes && <aside className="deck-notes mt-8 rounded-lg border bg-muted p-4 text-sm">{slide.notes}</aside>}
    </section>)}
    <nav aria-label="슬라이드 탐색" className="deck-controls mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 p-6">
      <Button variant="outline" onClick={() => move(index - 1)} disabled={index === 0}>이전</Button>
      <label className="text-sm">슬라이드 <select aria-label="슬라이드 선택" className="mx-2 rounded border bg-background p-2" value={index} onChange={(event) => move(Number(event.target.value))}>{slides.map((slide, i) => <option key={i} value={i}>{i + 1}. {slide.title}</option>)}</select> / {slides.length}</label>
      <Button onClick={() => move(index + 1)} disabled={index === slides.length - 1}>다음</Button>
    </nav>
    <p className="deck-controls pb-6 text-center text-xs text-muted-foreground">방향키 · Home / End · 링크의 #번호로 슬라이드 공유</p>
  </div>;
}
