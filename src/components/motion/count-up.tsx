'use client';

import { useEffect, useRef } from 'react';
import { useSiteMotion } from './motion-provider';

export function CountUp({ value }: { value: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const seen = useRef(false);
  const { canAnimate } = useSiteMotion();
  const formatted = value.toLocaleString('en-US');
  useEffect(() => {
    const node = ref.current;
    if (!node || !canAnimate || seen.current || !('IntersectionObserver' in window)) return;
    let frame = 0;
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return;
      seen.current = true; observer.disconnect();
      const start = performance.now();
      const tick = (now: number) => {
        const progress = Math.min(1, (now - start) / 800);
        node.textContent = Math.round(value * (1 - (1 - progress) ** 3)).toLocaleString('en-US');
        if (progress < 1) frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
    }, { threshold: .5 });
    observer.observe(node);
    return () => { observer.disconnect(); cancelAnimationFrame(frame); node.textContent = formatted; };
  }, [canAnimate, value, formatted]);
  return <span className="tabular-nums"><span className="sr-only">{formatted}</span><span ref={ref} aria-hidden="true" data-motion-count={value}>{formatted}</span></span>;
}
