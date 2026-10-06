'use client';

import { useLanguage } from '@/components/language-provider';
import { useState, useMemo, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Search, Calendar, MapPin, ExternalLink, Clock, Star, Download, RotateCcw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { conferenceCalendar, deadlineInstant, deadlineKind, isUpcoming, submissionStatus, upcomingDeadlines, type DeadlineScope } from '@/lib/conference-deadlines';
import { conferenceDataUpdated, legacyConferenceVerification, type Conference, type ConferenceCategory, type ConferenceCategoryInfo, type ConferenceDeadline } from '@/data/conferences';

const SAVED_KEY = 'suanlab-saved-conferences';

function useNow() {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const id = setInterval(() => setNow(new Date()), 60_000);
    return () => clearInterval(id);
  }, []);
  return now;
}

function countdown(target: Date, now: Date, ko: boolean) {
  const minutes = Math.max(0, Math.ceil((target.getTime() - now.getTime()) / 60_000));
  const days = Math.floor(minutes / 1440), hours = Math.floor(minutes % 1440 / 60);
  if (days) return ko ? `${days}일 ${hours}시간 남음` : `${days}d ${hours}h left`;
  return ko ? `${hours}시간 ${minutes % 60}분 남음` : `${hours}h ${minutes % 60}m left`;
}

export default function DeadlinesClient({ conferences, categories }: { conferences: Conference[]; categories: ConferenceCategoryInfo[] }) {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [selectedCategories, setSelectedCategories] = useState<Set<ConferenceCategory>>(new Set());
  const [query, setQuery] = useState('');
  const [year, setYear] = useState('all');
  const [scope, setScope] = useState<DeadlineScope>('submission');
  const [showPassed, setShowPassed] = useState(false);
  const [urgentOnly, setUrgentOnly] = useState(false);
  const [savedOnly, setSavedOnly] = useState(false);
  const [saved, setSaved] = useState<string[]>([]);
  const [timezone, setTimezone] = useState('Asia/Seoul');
  const now = useNow();

  useEffect(() => {
    try {
      const value: unknown = JSON.parse(localStorage.getItem(SAVED_KEY) ?? '[]');
      if (Array.isArray(value)) setSaved(value.filter((id): id is string => typeof id === 'string' && conferences.some(c => c.id === id)));
    } catch { /* Storage is optional. */ }
  }, [conferences]);

  function toggleSaved(id: string) {
    const next = saved.includes(id) ? saved.filter(x => x !== id) : [...saved, id];
    setSaved(next);
    try { localStorage.setItem(SAVED_KEY, JSON.stringify(next)); } catch { /* Storage is optional. */ }
  }
  function resetFilters() { setQuery(''); setSelectedCategories(new Set()); setYear('all'); setUrgentOnly(false); setSavedOnly(false); setShowPassed(false); }

  function displayDate(d: ConferenceDeadline, conf: Conference) {
    if (!d.date) return ko ? '발표 예정' : 'To be announced';
    if (deadlineKind(d) === 'event') return `${d.date}${d.endDate ? ` – ${d.endDate}` : ''}`;
    const instant = deadlineInstant(d, conf);
    if (instant && timezone !== 'source') return new Intl.DateTimeFormat(ko ? 'ko-KR' : 'en-GB', {
      timeZone: timezone === 'local' ? undefined : timezone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZoneName: 'short',
    }).format(instant);
    return `${d.date}${instant ? ` ${d.time ?? conf.deadlineTime ?? '23:59:59'} ${d.timezone ?? conf.timezone}` : ''}`;
  }

  const urgent = (conf: Conference) => {
    if (!now) return false;
    const next = upcomingDeadlines(conf, now, 'submission')[0];
    if (!next?.date) return false;
    const instant = deadlineInstant(next, conf) ?? new Date(`${next.date}T23:59:59Z`);
    return instant.getTime() - now.getTime() <= 30 * 86400_000;
  };
  const filtered = useMemo(() => {
    const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return conferences.filter(conf => {
      if (year !== 'all' && conf.year !== Number(year)) return false;
      if (selectedCategories.size && !conf.categories.some(c => selectedCategories.has(c))) return false;
      if (savedOnly && !saved.includes(conf.id)) return false;
      if (urgentOnly && !urgent(conf)) return false;
      const hay = `${conf.name} ${conf.full_name} ${conf.location} ${conf.year} ${conf.categories.join(' ')} ${conf.deadlines.map(d => d.type).join(' ')}`.toLowerCase();
      if (!terms.every(term => hay.includes(term))) return false;
      if (now && !showPassed) {
        if (scope === 'submission' && submissionStatus(conf, now) === 'closed') return false;
        if (scope === 'schedule' && !upcomingDeadlines(conf, now, scope).length && !conf.deadlines.some(d => !d.date)) return false;
      }
      return true;
    }).sort((a, b) => {
      if (!now) return a.name.localeCompare(b.name);
      const nextA = upcomingDeadlines(a, now, scope)[0], nextB = upcomingDeadlines(b, now, scope)[0];
      const rank = (c: Conference, d?: ConferenceDeadline) => d?.date ? (deadlineInstant(d, c)?.getTime() ?? Date.parse(`${d.date}T23:59:59Z`)) : Number.MAX_SAFE_INTEGER;
      return rank(a, nextA) - rank(b, nextB) || a.name.localeCompare(b.name);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conferences, selectedCategories, query, year, scope, showPassed, urgentOnly, savedOnly, saved, now]);
  const calendarCount = now ? filtered.reduce((sum, conf) => sum + upcomingDeadlines(conf, now, scope).length, 0) : 0;

  function downloadCalendar(items: Conference[]) {
    if (!now) return;
    const url = URL.createObjectURL(new Blob([conferenceCalendar(items, now, scope)], { type: 'text/calendar;charset=utf-8' }));
    const a = document.createElement('a'); a.href = url; a.download = 'suanlab-conference-deadlines.ics'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <>
    <div className="mb-8 rounded-xl border bg-muted/30 p-5">
      <p className="text-xs font-semibold uppercase tracking-wider text-primary">Research Calendar · {conferenceDataUpdated}</p>
      <h2 className="mt-2 text-xl font-semibold">{ko ? '다음 투고를 준비하는 연구 일정' : 'Plan your next research submission'}</h2>
      <p className="mt-2 text-sm leading-6 text-muted-foreground">{ko ? '투고 마감과 리뷰·발표·개최 일정을 구분합니다. 한국 시간으로 확인하고 관심 학회를 저장하거나 캘린더로 가져가세요.' : 'Separate submission deadlines from reviews, decisions and conference dates. Convert timezones, save venues and export your calendar.'}</p>
      <p className="mt-2 text-xs text-muted-foreground">{ko ? '초록 등록·ARR 리뷰 등 선행 조건은 각 학회 CFP를 확인하세요. 학회별 공식 출처와 확인일을 표시합니다.' : 'Check each CFP for prerequisites such as abstract registration or ARR reviews. Official sources and verification dates are shown per venue.'}</p>
    </div>
    <div className="mb-8 grid grid-cols-2 gap-3 md:grid-cols-4">
      {[
        [conferences.length, ko ? '수록 학회' : 'Venues'],
        [now ? conferences.filter(c => submissionStatus(c, now) === 'upcoming').length : '—', ko ? '투고 마감 예정' : 'Upcoming submissions'],
        [now ? conferences.filter(urgent).length : '—', ko ? '30일 이내 마감' : 'Due within 30 days'],
        [now ? conferences.filter(c => submissionStatus(c, now) === 'tba').length : '—', ko ? '투고 일정 미정' : 'Submission dates TBA'],
      ].map(([value, label]) => <Card key={label}><CardContent className="p-4"><p className="text-2xl font-bold text-primary">{value}</p><p className="mt-1 text-xs text-muted-foreground">{label}</p></CardContent></Card>)}
    </div>
    <div className="mb-6 space-y-4 rounded-xl border p-4">
      <div className="flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1"><Search aria-hidden="true" className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" /><Input aria-label={ko ? '학회 검색' : 'Search conferences'} placeholder={ko ? '학회·장소·트랙 검색 (예: SIGMOD 2027)' : 'Search venue, location or track (e.g. SIGMOD 2027)'} value={query} onChange={e => setQuery(e.target.value)} className="pl-9" /></div>
        <select aria-label={ko ? '학회 연도' : 'Conference year'} value={year} onChange={e => setYear(e.target.value)} className="h-10 rounded-md border bg-background px-3 text-sm"><option value="all">{ko ? '전체 연도' : 'All years'}</option>{Array.from(new Set(conferences.map(c => c.year))).sort().map(y => <option key={y} value={y}>{y}</option>)}</select>
        <select aria-label={ko ? '표시 시간대' : 'Display timezone'} value={timezone} onChange={e => setTimezone(e.target.value)} className="h-10 rounded-md border bg-background px-3 text-sm"><option value="Asia/Seoul">{ko ? '한국 시간 (KST)' : 'Korea (KST)'}</option><option value="local">{ko ? '내 시간대' : 'My timezone'}</option><option value="source">{ko ? '학회 원본 시간대' : 'Original timezone'}</option></select>
      </div>
      <div className="flex flex-wrap gap-2">{categories.map(cat => <Button key={cat.id} size="sm" variant={selectedCategories.has(cat.id) ? 'default' : 'outline'} aria-pressed={selectedCategories.has(cat.id)} onClick={() => setSelectedCategories(prev => { const next = new Set(prev); if (next.has(cat.id)) next.delete(cat.id); else next.add(cat.id); return next; })}>{cat.label}</Button>)}</div>
      <div className="flex flex-wrap gap-2 border-t pt-4">
        <Button size="sm" variant={scope === 'submission' ? 'default' : 'outline'} aria-pressed={scope === 'submission'} onClick={() => setScope('submission')}>{ko ? '투고 마감' : 'Submissions'}</Button>
        <Button size="sm" variant={scope === 'schedule' ? 'default' : 'outline'} aria-pressed={scope === 'schedule'} onClick={() => setScope('schedule')}>{ko ? '전체 일정' : 'All milestones'}</Button>
        <Button size="sm" variant={urgentOnly ? 'default' : 'outline'} aria-pressed={urgentOnly} onClick={() => setUrgentOnly(v => !v)}>{ko ? '30일 이내' : 'Within 30 days'}</Button>
        <Button size="sm" variant={savedOnly ? 'default' : 'outline'} aria-pressed={savedOnly} onClick={() => setSavedOnly(v => !v)}><Star aria-hidden="true" className="mr-1 h-3.5 w-3.5" />{ko ? '관심 학회' : 'Saved'} ({saved.length})</Button>
        <Button size="sm" variant={showPassed ? 'default' : 'outline'} aria-pressed={showPassed} onClick={() => setShowPassed(v => !v)}>{ko ? '지난 마감 포함' : 'Include past'}</Button>
        <Button size="sm" variant="ghost" onClick={resetFilters}><RotateCcw aria-hidden="true" className="mr-1 h-3.5 w-3.5" />{ko ? '초기화' : 'Reset'}</Button>
      </div>
    </div>
    <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
      <p role="status" className="text-sm text-muted-foreground">{filtered.length}{ko ? '개 학회 · 마감이 가까운 순' : ' venues · nearest deadline first'}</p>
      <Button size="sm" variant="outline" disabled={!calendarCount} onClick={() => downloadCalendar(filtered)}><Download aria-hidden="true" className="mr-2 h-4 w-4" />{ko ? '필터된 일정 다운로드' : 'Download filtered calendar'} ({calendarCount})</Button>
    </div>
    <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
      {filtered.map(conf => {
        const status = now ? submissionStatus(conf, now) : null;
        const next = now ? upcomingDeadlines(conf, now, scope)[0] : undefined;
        const instant = next ? deadlineInstant(next, conf) : null;
        return <Card key={conf.id} data-conference={conf.id} className="flex h-full flex-col">
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0"><CardTitle className="text-lg"><a href={conf.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 hover:text-primary">{conf.name} {conf.year}<ExternalLink aria-hidden="true" className="h-3 w-3 shrink-0" /></a></CardTitle><p className="mt-2 text-xs leading-5 text-muted-foreground">{conf.full_name}</p></div>
              <button type="button" onClick={() => toggleSaved(conf.id)} aria-label={`${ko ? '관심 학회 저장' : 'Save venue'}: ${conf.name} ${conf.year}`} aria-pressed={saved.includes(conf.id)} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md border hover:bg-accent"><Star aria-hidden="true" className={cn('h-4 w-4', saved.includes(conf.id) && 'fill-primary text-primary')} /></button>
            </div>
            <div className="flex flex-wrap gap-1.5 pt-2">{conf.categories.map(id => <Badge key={id} variant="secondary" className={categories.find(c => c.id === id)?.color}>{id}</Badge>)}</div>
          </CardHeader>
          <CardContent className="flex flex-1 flex-col gap-4">
            <p className="flex items-start gap-2 text-xs text-muted-foreground"><MapPin aria-hidden="true" className="h-4 w-4 shrink-0" />{conf.location}</p>
            <Badge variant="outline" className="self-start">{!status ? (ko ? '일정 확인 중' : 'Loading schedule') : status === 'upcoming' ? (ko ? '투고 마감 예정' : 'Submission upcoming') : status === 'tba' ? (ko ? '투고 일정 미정' : 'Submission TBA') : (ko ? '투고 마감' : 'Submission closed')}</Badge>
            {next ? <div className="rounded-lg border border-primary/25 bg-primary/5 p-3">
              <p className="text-xs font-semibold">{next.type}</p><p className="mt-2 break-words text-sm font-medium">{displayDate(next, conf)}</p>
              {instant && now ? <p className="mt-2 flex items-center gap-1.5 text-xs font-medium text-primary"><Clock aria-hidden="true" className="h-3.5 w-3.5" />{countdown(instant, now, ko)}</p> : <p className="mt-2 text-xs text-muted-foreground">{deadlineKind(next) === 'event' ? (ko ? '개최 일정' : 'Conference dates') : (ko ? '마감 시각은 공식 출처 확인' : 'Check the official source for the exact time')}</p>}
              {next.note && <p className="mt-2 text-xs leading-5 text-muted-foreground">{next.note}</p>}
            </div> : <p className="text-sm leading-6 text-muted-foreground">{status === 'tba' ? (ko ? '확정된 투고 날짜가 아직 없습니다. 공식 발표를 기다리고 있습니다.' : 'Confirmed submission dates are not yet available. Awaiting the official announcement.') : (ko ? '다가오는 마감이 없습니다. 전체 일정에서 후속 일정을 확인하세요.' : 'No upcoming deadlines in this view. See all milestones for follow-up dates.')}</p>}
            <details className="mt-auto rounded-md border p-3"><summary className="cursor-pointer text-xs font-medium">{ko ? '학회 전체 일정' : 'Full venue schedule'} ({conf.deadlines.length})</summary><ul className="mt-3 space-y-3">{conf.deadlines.map((d, i) => <li key={`${d.type}-${i}`} className={cn('text-xs', now && d.date && !isUpcoming(d, conf, now) && d.status !== 'tentative' && 'text-muted-foreground')}><p className="font-medium">{d.type}{d.status === 'tentative' && <span className="ml-2">{ko ? '(잠정)' : '(tentative)'}</span>}</p><p className="mt-1 break-words">{displayDate(d, conf)}</p>{d.note && <p className="mt-1 leading-5 text-muted-foreground">{d.note}</p>}</li>)}</ul></details>
            <div className="flex flex-wrap items-center justify-between gap-2 border-t pt-3"><a href={conf.sourceUrl ?? conf.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-xs text-primary hover:underline">{ko ? '공식 일정 / CFP' : 'Official dates / CFP'}<ExternalLink aria-hidden="true" className="h-3 w-3" /></a><span className="text-[11px] text-muted-foreground">{ko ? '확인' : 'Checked'} {conf.verifiedAt ?? legacyConferenceVerification}</span></div>
          </CardContent>
        </Card>;
      })}
    </div>
    {!filtered.length && <div className="rounded-xl border py-16 text-center"><Calendar aria-hidden="true" className="mx-auto mb-3 h-9 w-9 text-muted-foreground" /><p>{ko ? '조건에 맞는 학회가 없습니다.' : 'No matching venues.'}</p><Button variant="outline" className="mt-4" onClick={resetFilters}>{ko ? '필터 초기화' : 'Reset filters'}</Button></div>}
    <p className="mt-8 text-xs leading-6 text-muted-foreground">{ko ? '관심 학회는 이 브라우저에 저장됩니다. 캘린더에는 확정된 날짜의 예정 일정만 포함하며, 시각 미공개 일정은 종일 일정으로 내보냅니다. 시간대가 없는 항목은 AoE로 가정하지 않습니다. 일정 변경은 각 학회의 공식 출처를 확인하세요.' : 'Saved venues stay in this browser. Calendars include only upcoming confirmed dates; dates without a published time are exported as all-day events. Missing timezones are never assumed to be AoE. Consult the official sources for changes.'}</p>
  </>;
}
