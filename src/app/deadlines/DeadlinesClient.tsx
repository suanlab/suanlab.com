'use client';

import { useState, useMemo, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Search, Calendar, MapPin, ExternalLink, Clock, Globe, History } from 'lucide-react';
import { cn } from '@/lib/utils';
import type {
  Conference,
  ConferenceCategory,
  ConferenceCategoryInfo,
} from '@/data/conferences';

interface DeadlinesClientProps {
  conferences: Conference[];
  categories: ConferenceCategoryInfo[];
}

function useNow() {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const id = setInterval(() => setNow(new Date()), 60_000);
    return () => clearInterval(id);
  }, []);
  return now;
}

function formatCountdown(target: Date, now: Date | null): string {
  if (!now) return '';
  const diff = target.getTime() - now.getTime();
  if (diff <= 0) return '마감됨 / Passed';
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  if (days > 0) return `${days}일 ${hours}시간 남음`;
  if (hours > 0) return `${hours}시간 ${minutes}분 남음`;
  return `${minutes}분 남음`;
}

function toDateObj(date: string, timezone?: string): Date {
  if (timezone === 'UTC') return new Date(date + 'T23:59:59Z');
  return new Date(date + 'T23:59:59-12:00');
}

function getNextDeadline(conf: Conference, now: Date | null) {
  if (!now) return null;
  const upcoming = conf.deadlines
    .map((d) => ({ ...d, dateObj: toDateObj(d.date, conf.timezone) }))
    .filter((d) => d.dateObj.getTime() > now.getTime())
    .sort((a, b) => a.dateObj.getTime() - b.dateObj.getTime());
  return upcoming[0] ?? null;
}

function getLastDeadline(conf: Conference) {
  const sorted = [...conf.deadlines]
    .map((d) => ({ ...d, dateObj: toDateObj(d.date, conf.timezone) }))
    .sort((a, b) => b.dateObj.getTime() - a.dateObj.getTime());
  return sorted[0] ?? null;
}

export default function DeadlinesClient({ conferences, categories }: DeadlinesClientProps) {
  const [selectedCategories, setSelectedCategories] = useState<Set<ConferenceCategory>>(new Set());
  const [query, setQuery] = useState('');
  const [showPassed, setShowPassed] = useState(false);
  const now = useNow();

  const toggleCategory = (cat: ConferenceCategory) => {
    setSelectedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  };

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return conferences
      .filter((conf) => {
        if (selectedCategories.size > 0) {
          const hasAny = conf.categories.some((c) => selectedCategories.has(c));
          if (!hasAny) return false;
        }
        if (q) {
          const hay = `${conf.name} ${conf.full_name} ${conf.location} ${conf.year}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }
        if (!showPassed && now) {
          const last = getLastDeadline(conf);
          if (last && last.dateObj.getTime() < now.getTime()) return false;
        }
        return true;
      })
      .sort((a, b) => {
        if (!now) return a.name.localeCompare(b.name);
        const aNext = getNextDeadline(a, now);
        const bNext = getNextDeadline(b, now);
        if (!aNext && !bNext) return 0;
        if (!aNext) return 1;
        if (!bNext) return -1;
        return aNext.dateObj.getTime() - bNext.dateObj.getTime();
      });
  }, [conferences, selectedCategories, query, showPassed, now]);

  const activeCount = useMemo(() => {
    if (!now) return 0;
    return conferences.filter((conf) => getNextDeadline(conf, now) !== null).length;
  }, [conferences, now]);

  return (
    <>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-primary">{conferences.length}</p>
            <p className="text-sm text-muted-foreground">전체 학회</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-green-600 dark:text-green-400">{activeCount}</p>
            <p className="text-sm text-muted-foreground">진행 중</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-muted-foreground">{categories.length}</p>
            <p className="text-sm text-muted-foreground">분야</p>
          </CardContent>
        </Card>
      </div>

      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="학회 이름, 장소 검색..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        {categories.map((cat) => {
          const active = selectedCategories.has(cat.id);
          return (
            <Button
              key={cat.id}
              variant={active ? 'default' : 'outline'}
              size="sm"
              onClick={() => toggleCategory(cat.id)}
              className={cn(!active && 'hover:bg-accent')}
            >
              {cat.label}
            </Button>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 mb-6">
        <Badge variant="outline" className="text-sm">
          {filtered.length}개 표시 중
        </Badge>
        <Button
          variant={showPassed ? 'default' : 'ghost'}
          size="sm"
          onClick={() => setShowPassed((v) => !v)}
        >
          <History className="mr-1 h-3 w-3" />
          {showPassed ? '마감된 학회 표시 중' : '마감된 학회 숨김'}
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {filtered.map((conf) => {
          const next = getNextDeadline(conf, now);
          const passed = !next;
          return (
            <Card
              key={conf.id}
              className={cn(
                'h-full flex flex-col transition-all hover:shadow-md',
                passed && 'opacity-60'
              )}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-2 mb-1">
                  <CardTitle className="text-lg leading-tight">
                    <a
                      href={conf.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 hover:text-primary"
                    >
                      {conf.name} {conf.year}
                      <ExternalLink className="h-3 w-3 opacity-60" />
                    </a>
                  </CardTitle>
                  {next && (
                    <Badge className="bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300 shrink-0">
                      <Clock className="mr-1 h-3 w-3" />
                      {formatCountdown(next.dateObj, now)}
                    </Badge>
                  )}
                  {passed && <Badge variant="secondary">마감</Badge>}
                </div>
                <p className="text-xs text-muted-foreground">{conf.full_name}</p>
                <div className="flex flex-wrap gap-1 mt-2">
                  {conf.categories.map((catId) => {
                    const info = categories.find((c) => c.id === catId);
                    if (!info) return null;
                    return (
                      <span
                        key={catId}
                        className={cn(
                          'inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium',
                          info.color
                        )}
                      >
                        {info.label}
                      </span>
                    );
                  })}
                </div>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <div className="space-y-1.5 text-sm text-muted-foreground mb-3">
                  <div className="flex items-center gap-2">
                    <MapPin className="h-4 w-4 shrink-0" />
                    <span>{conf.location}</span>
                  </div>
                  {conf.timezone && (
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 shrink-0" />
                      <span className="text-xs">
                        {conf.timezone === 'UTC' ? 'UTC 기준' : 'AoE (UTC-12) 기준'}
                      </span>
                    </div>
                  )}
                </div>

                <div className="mt-auto space-y-1.5">
                  <p className="text-xs font-medium text-muted-foreground">일정</p>
                  <ul className="space-y-1">
                    {conf.deadlines.map((d, idx) => {
                      const dObj = toDateObj(d.date, conf.timezone);
                      const isNext = next && d.type === next.type && d.date === next.date;
                      const isPassed = now ? dObj.getTime() < now.getTime() : false;
                      return (
                        <li
                          key={idx}
                          className={cn(
                            'flex items-center justify-between text-xs',
                            isNext && 'font-medium text-primary',
                            isPassed && !isNext && 'text-muted-foreground/60 line-through'
                          )}
                        >
                          <span>{d.type}</span>
                          <span className="font-mono text-right">
                            {d.date}
                            {d.note && (
                              <span className="block text-[10px] text-muted-foreground font-sans">
                                {d.note}
                              </span>
                            )}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </div>

                <div className="mt-3 flex items-center gap-2">
                  <Button asChild size="sm" variant="outline">
                    <a href={conf.url} target="_blank" rel="noopener noreferrer">
                      <Globe className="mr-1 h-3 w-3" />
                      웹사이트
                    </a>
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-16 text-muted-foreground">
          <Calendar className="h-10 w-10 mx-auto mb-3 opacity-40" />
          <p>조건에 맞는 학회가 없습니다.</p>
        </div>
      )}

      <p className="mt-10 text-xs text-muted-foreground text-center">
        마감일은 학회별 표기 시간대(AoE/UTC)를 따르며 변경될 수 있습니다.
        <br />
        정확한 일정은 각 학회 공식 웹사이트를 반드시 확인하세요.
      </p>
    </>
  );
}
