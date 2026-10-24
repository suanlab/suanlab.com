'use client';
import { useEffect, useState, type RefObject } from 'react';
import { useSiteMotion } from './motion-provider';

/** Decorative loops run only while their scene is visible and motion is enabled. */
export function useSceneActivity<T extends Element>(ref: RefObject<T>) {
  const { canAnimate } = useSiteMotion();
  const [active, setActive] = useState(false);
  useEffect(() => {
    if (!canAnimate || !ref.current || !('IntersectionObserver' in window)) return;
    let inView = false;
    const update = () => setActive(inView && !document.hidden);
    const observer = new IntersectionObserver(([entry]) => { inView = entry.isIntersecting; update(); });
    observer.observe(ref.current); document.addEventListener('visibilitychange', update);
    return () => { observer.disconnect(); document.removeEventListener('visibilitychange', update); setActive(false); };
  }, [canAnimate, ref]);
  return canAnimate && active;
}
