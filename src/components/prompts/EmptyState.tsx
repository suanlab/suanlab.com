'use client';

import { Search } from 'lucide-react';


export default function EmptyState({ lang }: { lang: 'ko' | 'en' }) {
  return (
    <div className="text-center py-16 text-muted-foreground col-span-full">
      <Search className="h-10 w-10 mx-auto mb-3 opacity-40" />
      <p>{lang === 'ko' ? '조건에 맞는 프롬프트가 없습니다.' : 'No prompts match your search.'}</p>
    </div>
  );
}
