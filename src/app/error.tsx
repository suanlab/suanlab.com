'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { AlertTriangle, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function Error({ error, reset }: ErrorProps) {
  useEffect(() => {
    // Log error to console for debugging
    console.error('Application error:', error);
  }, [error]);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-[url('/assets/images/slider/2.jpg')] bg-cover bg-center opacity-10" />
      <div className="absolute inset-0 bg-gradient-to-r from-red-500/10 to-transparent" />

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute w-1 h-1 bg-red-400 rounded-full animate-particle-1 opacity-60" style={{ left: '10%', top: '20%' }} />
        <div className="absolute w-1.5 h-1.5 bg-orange-400 rounded-full animate-particle-2 opacity-50" style={{ left: '80%', top: '30%' }} />
        <div className="absolute w-1 h-1 bg-red-500 rounded-full animate-particle-3 opacity-70" style={{ left: '20%', top: '60%' }} />
        <div className="absolute w-2 h-2 bg-orange-300 rounded-full animate-particle-4 opacity-40" style={{ left: '70%', top: '70%' }} />
      </div>

      {/* Content */}
      <div className="container relative z-10">
        <div className="mx-auto max-w-2xl text-center">
          {/* Error Icon */}
          <div className="mb-8 flex justify-center">
            <div className="rounded-full bg-red-500/20 p-6 border border-red-500/30">
              <AlertTriangle className="h-16 w-16 text-red-400" />
            </div>
          </div>

          {/* Title */}
          <h1 className="text-3xl md:text-4xl font-bold mb-4">
            오류가 발생했습니다
          </h1>

          {/* Description */}
          <p className="text-lg text-slate-300 mb-2">
            예상치 못한 오류가 발생했습니다. 잠시 후 다시 시도해주세요.
          </p>

          {/* Error Details (if available) */}
          {error.message && (
            <div className="mt-6 p-4 rounded-lg bg-slate-800/50 border border-slate-700 text-left">
              <p className="text-sm text-slate-400 font-mono break-words">
                {error.message}
              </p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="mt-8 flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" onClick={reset}>
              <AlertTriangle className="mr-2 h-4 w-4" />
              다시 시도
            </Button>
            <Button size="lg" variant="outline" className="bg-white/10 hover:bg-white/20" asChild>
              <Link href="/">
                <Home className="mr-2 h-4 w-4" />
                홈으로 돌아가기
              </Link>
            </Button>
          </div>

          {/* Tech Accent Line */}
          <div className="mt-12 flex items-center justify-center gap-2">
            <div className="h-0.5 w-2 bg-red-400 rounded-full shadow-[0_0_8px_rgba(248,113,113,0.6)] animate-pulse" />
            <div className="h-0.5 w-8 bg-gradient-to-r from-red-400 to-orange-500 rounded-full shadow-[0_0_10px_rgba(248,113,113,0.5)] animate-pulse animation-delay-2000" />
            <div className="h-0.5 w-16 bg-gradient-to-r from-orange-500 to-red-500 rounded-full shadow-[0_0_12px_rgba(249,115,22,0.4)] animate-pulse animation-delay-4000" />
          </div>
        </div>
      </div>
    </section>
  );
}
