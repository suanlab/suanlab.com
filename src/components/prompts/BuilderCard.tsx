'use client';
import { englishText } from '@/data/i18n/translate';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Star, Wand2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type PromptBuilder, type LocalizedText } from '@/data/prompts';


export default function BuilderCard({
  builder, Icon, isFav, onToggleFav, onOpen, catLabel, catColor, L, lang,
}: {
  builder: PromptBuilder;
  Icon: React.ComponentType<{ className?: string }>;
  isFav: boolean;
  onToggleFav: () => void;
  onOpen: () => void;
  catLabel: string;
  catColor: string;
  L: (t: LocalizedText) => string;
  lang: 'ko' | 'en';
}) {
  return (
    <Card className="h-full flex flex-col transition-all hover:shadow-md hover:-translate-y-0.5 group">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <div className={cn('inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary shrink-0')}><Icon className="h-5 w-5" /></div>
          <button onClick={(e) => { e.stopPropagation(); onToggleFav(); }} aria-pressed={isFav} aria-label={lang === 'ko' ? '즐겨찾기 변경' : 'Toggle favorite'} className="text-muted-foreground hover:text-amber-500 transition-colors">
            <Star className={cn('h-5 w-5', isFav && 'fill-amber-400 text-amber-400')} />
          </button>
        </div>
        <CardTitle className="text-base mt-2 group-hover:text-primary transition-colors">{L(builder.title)}</CardTitle>
        <p className="text-sm text-muted-foreground line-clamp-2">{L(builder.description)}</p>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col">
        <div className="flex flex-wrap gap-1.5 mb-3">
          <span className={cn('inline-flex items-center rounded px-2 py-0.5 text-[10px] font-medium', catColor)}>{catLabel}</span>
          {builder.tags.slice(0, 2).map((t) => (
            <span key={t} className="inline-flex items-center rounded bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">#{lang === 'en' ? englishText(t) : t}</span>
          ))}
        </div>
        <Button className="mt-auto w-full" size="sm" onClick={onOpen}>
          <Wand2 className="mr-1.5 h-4 w-4" />
          {lang === 'ko' ? '생성하기' : 'Open builder'}
        </Button>
      </CardContent>
    </Card>
  );
}
