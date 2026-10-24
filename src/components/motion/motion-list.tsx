'use client';

import { useId, type ReactNode } from 'react';
import { LazyMotion, LayoutGroup, MotionConfig } from 'motion/react';
import * as m from 'motion/react-m';
import { useSiteMotion } from './motion-provider';

const loadFeatures = () => import('./features').then(module => module.default);

export function MotionList({ children, className }: { children: ReactNode; className?: string }) {
  const id = useId();
  const { canAnimate } = useSiteMotion();
  return <LazyMotion features={loadFeatures} strict>
    <MotionConfig reducedMotion={canAnimate ? 'user' : 'always'}>
      <LayoutGroup id={id}><div data-motion-list className={className}>{children}</div></LayoutGroup>
    </MotionConfig>
  </LazyMotion>;
}

export function MotionItem({ children, className }: { children: ReactNode; className?: string }) {
  const { canAnimate } = useSiteMotion();
  return <m.div data-motion-item layout={canAnimate ? 'position' : false} initial={false}
    transition={{ layout: { duration: .24, ease: [.2, .7, .2, 1] } }} className={className}>{children}</m.div>;
}
