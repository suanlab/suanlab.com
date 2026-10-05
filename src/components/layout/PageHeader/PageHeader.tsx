'use client';

import Link from 'next/link';
import { ChevronRight, Home } from 'lucide-react';
import { useLanguage } from '@/components/language-provider';
import MenuDotField from './MenuDotField';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  subtitleKey?: string;
  backgroundImage?: string;
  breadcrumbs?: { label: string; href?: string }[];
}

export default function PageHeader({ title, subtitle, subtitleKey, backgroundImage = '/assets/images/slider/blue.jpg', breadcrumbs = [] }: PageHeaderProps) {
  const { t, language } = useLanguage();
  const displaySubtitle = subtitleKey ? (t(subtitleKey) as string) : subtitle;
  return (
    <section className="menu-page-header relative overflow-hidden py-14 text-white md:py-16 lg:py-20" style={{ backgroundImage: `url(${backgroundImage})`, backgroundSize: 'cover', backgroundPosition: 'center' }}>
      <div aria-hidden="true" className="absolute inset-0 bg-gradient-to-br from-slate-950/95 via-slate-900/90 to-blue-950/80" />
      <MenuDotField />
      <div className="container relative z-10">
        <div className="min-w-0 max-w-3xl">
          {breadcrumbs.length > 0 && (
            <nav aria-label={language === 'ko' ? '현재 위치' : 'Breadcrumbs'} className="mb-6 inline-flex max-w-full flex-wrap items-center gap-y-2 rounded-lg border border-white/15 bg-slate-950/65 px-4 py-2 text-sm">
              <Link href="/" className="flex items-center text-cyan-200 hover:text-white"><Home aria-hidden="true" className="h-4 w-4" /><span className="sr-only">{language === 'ko' ? '홈' : 'Home'}</span></Link>
              {breadcrumbs.map((crumb, index) => (
                <span key={index} className="flex min-w-0 items-center">
                  <ChevronRight aria-hidden="true" className="mx-2 h-4 w-4 shrink-0 text-slate-300" />
                  {crumb.href ? <Link href={crumb.href} className="break-words text-slate-200 hover:text-cyan-100">{crumb.label}</Link> : <span aria-current="page" className="break-words font-medium text-cyan-100">{crumb.label}</span>}
                </span>
              ))}
            </nav>
          )}
          <h1 className="break-words text-3xl font-bold tracking-tight [overflow-wrap:anywhere] sm:text-4xl md:text-5xl lg:text-6xl">{title}</h1>
          {displaySubtitle && <p className="mt-4 max-w-2xl text-lg leading-relaxed text-slate-200 md:text-xl">{displaySubtitle}</p>}
          <div aria-hidden="true" className="mt-6 flex gap-2">{[0, 1, 2, 3, 4].map(i => <span key={i} className="h-1 w-1 rounded-full bg-cyan-200" style={{ opacity: 1 - i * .15 }} />)}</div>
        </div>
      </div>
    </section>
  );
}
