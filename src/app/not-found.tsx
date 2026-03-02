import Link from 'next/link';
import { Metadata } from 'next';
import { ArrowLeft, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const metadata: Metadata = {
  title: '404 - 페이지를 찾을 수 없습니다',
  description: '요청하신 페이지를 찾을 수 없습니다.',
};

export default function NotFound() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-[url('/assets/images/slider/2.jpg')] bg-cover bg-center opacity-10" />
      <div className="absolute inset-0 bg-gradient-to-r from-primary/10 to-transparent" />

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute w-1 h-1 bg-cyan-400 rounded-full animate-particle-1 opacity-60" style={{ left: '10%', top: '20%' }} />
        <div className="absolute w-1.5 h-1.5 bg-blue-400 rounded-full animate-particle-2 opacity-50" style={{ left: '80%', top: '30%' }} />
        <div className="absolute w-1 h-1 bg-purple-400 rounded-full animate-particle-3 opacity-70" style={{ left: '20%', top: '60%' }} />
        <div className="absolute w-2 h-2 bg-cyan-300 rounded-full animate-particle-4 opacity-40" style={{ left: '70%', top: '70%' }} />
      </div>

      {/* Content */}
      <div className="container relative z-10">
        <div className="mx-auto max-w-2xl text-center">
          {/* 404 Display */}
          <div className="mb-8">
            <h1 className="text-9xl md:text-[150px] font-bold tracking-tight">
              <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent drop-shadow-[0_0_30px_rgba(6,182,212,0.3)]">
                404
              </span>
            </h1>
          </div>

          {/* Title */}
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            페이지를 찾을 수 없습니다
          </h2>

          {/* Description */}
          <p className="text-lg text-slate-300 mb-8 max-w-xl mx-auto">
            요청하신 페이지가 존재하지 않거나 이동되었을 수 있습니다. 홈으로 돌아가거나 블로그를 확인해보세요.
          </p>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" asChild>
              <Link href="/">
                <Home className="mr-2 h-4 w-4" />
                홈으로 돌아가기
              </Link>
            </Button>
            <Button size="lg" variant="outline" className="bg-white/10 hover:bg-white/20" asChild>
              <Link href="/blog">
                <ArrowLeft className="mr-2 h-4 w-4" />
                블로그 보기
              </Link>
            </Button>
          </div>

          {/* Tech Accent Line */}
          <div className="mt-12 flex items-center justify-center gap-2">
            <div className="h-0.5 w-2 bg-cyan-400 rounded-full shadow-[0_0_8px_rgba(34,211,238,0.6)] animate-pulse" />
            <div className="h-0.5 w-8 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full shadow-[0_0_10px_rgba(6,182,212,0.5)] animate-pulse animation-delay-2000" />
            <div className="h-0.5 w-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full shadow-[0_0_12px_rgba(96,165,250,0.4)] animate-pulse animation-delay-4000" />
          </div>
        </div>
      </div>
    </section>
  );
}
