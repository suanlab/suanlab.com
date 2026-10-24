'use client';

import { useState, useMemo, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { promptsUpdatedAt, promptReferences } from '@/data/prompts/advanced';
import { readPromptTarget, normalizePromptValues, encodePromptShare as encodeShare, decodePromptShare as decodeShare } from '@/lib/prompt-state';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Search, Star, Wand2, Lightbulb, ShieldCheck, FileSearch, Compass, FlaskConical, PenLine, Target, MessageSquareReply, Library, FileText, Database, Presentation, GraduationCap, RotateCcw, ExternalLink, Plus, Workflow } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useLanguage } from '@/components/language-provider';
import { promptBuilders, promptSnippets, promptCategories, type PromptBuilder, type PromptSnippet, type PromptCategory, type LocalizedText, type PromptWorkflow } from '@/data/prompts';
import { promptWorkflows } from '@/data/prompts/workflows';
import BuilderCard from '@/components/prompts/BuilderCard';
import SnippetCard from '@/components/prompts/SnippetCard';
import EmptyState from '@/components/prompts/EmptyState';
import { prefillBuilder, customToSnippet } from '@/components/prompts/helpers';
import type { CustomSnippet } from '@/components/prompts/types';
import { loadWorkflowState, saveWorkflowOutput, resetWorkflowState } from '@/components/prompts/workflow-storage';

const BuilderDetail = dynamic(() => import('@/components/prompts/BuilderDetail'));
const CustomPromptForm = dynamic(() => import('@/components/prompts/CustomPromptForm'));

interface PromptsClientProps {
  workflows?: PromptWorkflow[];
}

const ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  lightbulb: Lightbulb,
  'shield-check': ShieldCheck,
  'file-search': FileSearch,
  compass: Compass,
  'flask-conical': FlaskConical,
  'pen-line': PenLine,
  target: Target,
  'message-square-reply': MessageSquareReply,
  library: Library,
  'file-text': FileText,
  database: Database,
  presentation: Presentation,
  'graduation-cap': GraduationCap,
};

const FAV_KEY = 'suanlab-prompt-favorites';
const CUSTOM_KEY = 'suanlab-custom-prompts';

export default function PromptsClient(_props: PromptsClientProps = {}) {
  const builders = promptBuilders;
  const baseSnippets = promptSnippets;
  const categories = promptCategories;
  const workflows = _props.workflows ?? promptWorkflows;
  const { language } = useLanguage();
  const L = useCallback((t: LocalizedText) => t[language], [language]);

  const [tab, setTab] = useState<'builders' | 'library' | 'workflows' | 'favorites'>('builders');
  const [query, setQuery] = useState('');
  const [selectedCats, setSelectedCats] = useState<Set<PromptCategory>>(new Set());
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());
  const [sort, setSort] = useState<'default' | 'alpha' | 'recent'>('default');
  const [showAllTags, setShowAllTags] = useState(false);
  const [, setWorkflowRevision] = useState(0);
  const [shareFailed, setShareFailed] = useState(false);
  const [workflowSaveFailed, setWorkflowSaveFailed] = useState(false);

  const [activeBuilder, setActiveBuilder] = useState<PromptBuilder | null>(null);
  const [builderVersion, setBuilderVersion] = useState(0);
  const [pendingValues, setPendingValues] = useState<Record<string, string | string[]> | null>(null);
  const [workflowCtx, setWorkflowCtx] = useState<{ wf: PromptWorkflow; step: number } | null>(null);

  const [favorites, setFavorites] = useState<string[]>([]);
  const [customs, setCustoms] = useState<CustomSnippet[]>([]);
  const [expandedSnip, setExpandedSnip] = useState<string | null>(null);
  const [snipValues, setSnipValues] = useState<Record<string, Record<string, string>>>({});
  const [showCustomForm, setShowCustomForm] = useState(false);
  const [editingCustom, setEditingCustom] = useState<CustomSnippet | null>(null);
  const [shared, setShared] = useState(false);
  const [linkedSnippet, setLinkedSnippet] = useState<string | null>(null);

  // combine custom snippets into the library (normalized to PromptSnippet)
  const customIds = useMemo(() => new Set(customs.map((c) => c.id)), [customs]);
  const snippets: PromptSnippet[] = useMemo(
    () => [...customs.map(customToSnippet), ...baseSnippets],
    [customs, baseSnippets],
  );

  // load from localStorage + parse share hash on mount
  useEffect(() => {
    try {
      const fav = localStorage.getItem(FAV_KEY);
      if (fav) { const value: unknown = JSON.parse(fav); if (Array.isArray(value)) setFavorites(value.filter((id): id is string => typeof id === 'string')); }
      const cus = localStorage.getItem(CUSTOM_KEY);
      if (cus) { const value: unknown = JSON.parse(cus); if (Array.isArray(value)) setCustoms(value.filter((c): c is CustomSnippet => !!c && typeof c.id === 'string' && typeof c.title === 'string' && typeof c.content === 'string' && Array.isArray(c.tags) && c.tags.every((tag: unknown) => typeof tag === 'string') && categories.some(cat => cat.id === c.category))); }
    } catch {
      /* noop */
    }
    const restoreTarget = () => {
      const decoded = decodeShare(window.location.hash);
      const target = readPromptTarget(new URLSearchParams(window.location.search));
      const builder = decoded ? builders.find(item => item.id === decoded.b)
        : target?.kind === 'builder' ? builders.find(item => item.id === target.id) : null;
      if (builder) {
        setTab('builders'); setActiveBuilder(builder); setWorkflowCtx(null);
        setPendingValues(decoded ? normalizePromptValues(builder.fields, decoded.v) : null);
        setBuilderVersion(version => version + 1);
      } else if (target?.kind === 'snippet' && baseSnippets.some(item => item.id === target.id)) {
        setTab('library'); setActiveBuilder(null); setWorkflowCtx(null); setPendingValues(null);
        setSelectedCats(new Set()); setSelectedTags(new Set()); setQuery('');
        setExpandedSnip(target.id); setLinkedSnippet(target.id);
      } else {
        setTab('builders'); setActiveBuilder(null); setWorkflowCtx(null); setPendingValues(null);
        setExpandedSnip(null); setLinkedSnippet(null);
      }
    };
    restoreTarget();
    window.addEventListener('hashchange', restoreTarget);
    window.addEventListener('popstate', restoreTarget);
    return () => { window.removeEventListener('hashchange', restoreTarget); window.removeEventListener('popstate', restoreTarget); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!linkedSnippet || tab !== 'library') return;
    const frame = requestAnimationFrame(() => {
      const card = document.getElementById(`prompt-${linkedSnippet}`);
      card?.scrollIntoView({ block: 'center' });
      card?.focus({ preventScroll: true });
      setLinkedSnippet(null);
    });
    return () => cancelAnimationFrame(frame);
  }, [linkedSnippet, tab]);

  function clearPromptTarget() {
    const url = new URL(window.location.href);
    url.searchParams.delete('builder'); url.searchParams.delete('snippet'); url.hash = '';
    window.history.replaceState(null, '', url);
  }

  const toggleFav = (id: string) => {
    setFavorites((prev) => {
      const next = prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id];
      try {
        localStorage.setItem(FAV_KEY, JSON.stringify(next));
      } catch {
        /* noop */
      }
      return next;
    });
  };

  const saveCustom = (c: CustomSnippet) => {
    setCustoms((prev) => {
      const exists = prev.some((x) => x.id === c.id);
      const next = exists ? prev.map((x) => (x.id === c.id ? c : x)) : [c, ...prev];
      try {
        localStorage.setItem(CUSTOM_KEY, JSON.stringify(next));
      } catch {
        /* noop */
      }
      return next;
    });
  };

  const deleteCustom = (id: string) => {
    if (favorites.includes(id)) toggleFav(id);
    setCustoms((prev) => {
      const next = prev.filter((x) => x.id !== id);
      try {
        localStorage.setItem(CUSTOM_KEY, JSON.stringify(next));
      } catch {
        /* noop */
      }
      return next;
    });
  };

  const toggleCat = (c: PromptCategory) => {
    setSelectedCats((prev) => {
      const next = new Set(prev);
      if (next.has(c)) next.delete(c);
      else next.add(c);
      return next;
    });
  };

  const toggleTag = (t: string) => {
    setSelectedTags((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });
  };

  // all tags (from builders + snippets)
  const allTags = useMemo(() => {
    const set = new Set<string>();
    builders.forEach((b) => b.tags.forEach((t) => set.add(t)));
    snippets.forEach((s) => s.tags.forEach((t) => set.add(t)));
    const featured = ['RAG', 'Agents', 'Multimodal', 'Evals', 'Text-to-SQL', 'LLMOps', 'Grounding'];
    return [...featured.filter(tag => set.has(tag)), ...[...set].filter(tag => !featured.includes(tag)).sort((a, b) => a.localeCompare(b))];
  }, [builders, snippets]);

  const q = query.trim().toLowerCase();
  const matchText = (hay: string) => !q || q.split(/\s+/).every(term => hay.toLowerCase().includes(term));
  const catOk = (c: PromptCategory) => selectedCats.size === 0 || selectedCats.has(c);
  const tagOk = (tags: string[]) => selectedTags.size === 0 || tags.some((t) => selectedTags.has(t));

  const sortFn = useCallback(
    <T extends { title: LocalizedText | string; updatedAt?: string }>(arr: T[]): T[] => {
      if (sort === 'recent') return [...arr].sort((a, b) => (b.updatedAt ?? '').localeCompare(a.updatedAt ?? ''));
      if (sort === 'alpha') return [...arr].sort((a, b) => L(a.title as LocalizedText).localeCompare(L(b.title as LocalizedText)));
      return arr;
    },
    [sort, L],
  );

  const filteredBuilders = useMemo(
    () =>
      sortFn(
        builders.filter(
          (b) => catOk(b.category) && tagOk(b.tags) && matchText(`${b.title.ko} ${b.title.en} ${b.description.ko} ${b.description.en} ${b.tags.join(' ')}`),
        ),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [builders, q, selectedCats, selectedTags, sort, language],
  );

  const filteredSnippets = useMemo(
    () =>
      sortFn(
        snippets.filter((s) => {
          const title = L(s.title);
          const desc = L(s.description);
          return catOk(s.category) && tagOk(s.tags) && matchText(`${title} ${desc} ${s.title.ko} ${s.title.en} ${s.description.ko} ${s.description.en} ${s.tags.join(' ')} ${s.content} ${s.contentEn ?? ''}`);
        }),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [snippets, q, selectedCats, selectedTags, sort, language],
  );

  const favBuilders = filteredBuilders.filter((b) => favorites.includes(b.id));
  const favSnippets = filteredSnippets.filter((s) => favorites.includes(s.id));

  const catLabel = (c: PromptCategory) => L(categories.find((x) => x.id === c)!.label);
  const catColor = (c: PromptCategory) => categories.find((x) => x.id === c)!.color;

  const handleShare = useCallback(
    (builderId: string, values: Record<string, string | string[]>) => {
      const encoded = encodeShare(builderId, values);
      if (!encoded) return;
      if (typeof window !== 'undefined') {
        const url = `${window.location.origin}/prompts/#${encoded}`;
        setShareFailed(false);
        if (!navigator.clipboard) { setShareFailed(true); return; }
        navigator.clipboard.writeText(url).then(
          () => {
            setShared(true);
            setTimeout(() => setShared(false), 2000);
          },
          () => setShareFailed(true),
        );
      }
    },
    [],
  );

  const openBuilder = (b: PromptBuilder, opts?: { values?: Record<string, string | string[]> | null; wf?: typeof workflowCtx }) => {
    setTab('builders');
    setShareFailed(false);
    setActiveBuilder(b);
    setBuilderVersion(v => v + 1);
    setPendingValues(opts?.values ?? null);
    setWorkflowCtx(opts?.wf ?? null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const renderToolbar = () => (
    <>
      <div className="relative mb-3">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          aria-label={language === 'ko' ? '프롬프트 검색' : 'Search prompts'}
          placeholder={language === 'ko' ? '제목·내용·태그 검색 (예: RAG 평가)' : 'Search titles, content or tags (e.g. RAG evaluation)'}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="pl-9"
        />
      </div>
      <div className="flex flex-wrap gap-2 mb-3">
        {categories.map((c) => {
          const active = selectedCats.has(c.id);
          return (
            <Button key={c.id} variant={active ? 'default' : 'outline'} size="sm" onClick={() => toggleCat(c.id)} aria-pressed={active} className={cn(!active && 'hover:bg-accent')}>
              {L(c.label)}
            </Button>
          );
        })}
        {(query || selectedCats.size > 0 || selectedTags.size > 0) && (
          <Button variant="ghost" size="sm" onClick={() => { setQuery(''); setSelectedCats(new Set()); setSelectedTags(new Set()); }}>
            <RotateCcw className="mr-1 h-3 w-3" />
            {language === 'ko' ? '초기화' : 'Reset'}
          </Button>
        )}
        <select
          aria-label={language === 'ko' ? '프롬프트 정렬' : 'Sort prompts'}
          value={sort}
          onChange={(e) => setSort(e.target.value as typeof sort)}
          className="ml-auto h-9 rounded-md border border-input bg-background px-2 text-xs"
        >
          <option value="default">{language === 'ko' ? '기본 순' : 'Default'}</option>
          <option value="recent">{language === 'ko' ? '최근 업데이트' : 'Recently updated'}</option>
          <option value="alpha">{language === 'ko' ? '가나다' : 'A→Z'}</option>
        </select>
      </div>
      {allTags.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-4">
          {(showAllTags ? allTags : allTags.slice(0, 12)).map((t) => {
            const on = selectedTags.has(t);
            return (
              <button key={t} type="button" onClick={() => toggleTag(t)} aria-pressed={on} className={cn('rounded-full border px-2 py-0.5 text-[11px] transition-colors', on ? 'border-primary bg-primary text-primary-foreground' : 'border-input bg-background hover:bg-accent text-muted-foreground')}>
                #{t}
              </button>
            );
          })}
          <button type="button" className="px-2 text-xs text-primary hover:underline" aria-expanded={showAllTags} onClick={() => setShowAllTags(v => !v)}>{language === 'ko' ? (showAllTags ? '태그 접기' : `모든 태그 (${allTags.length})`) : (showAllTags ? 'Fewer tags' : `All tags (${allTags.length})`)}</button>
        </div>
      )}
      <p role="status" className="mb-4 text-xs text-muted-foreground">{language === 'ko' ? `${tab === 'library' ? filteredSnippets.length : tab === 'favorites' ? favBuilders.length + favSnippets.length : filteredBuilders.length}개 결과` : `${tab === 'library' ? filteredSnippets.length : tab === 'favorites' ? favBuilders.length + favSnippets.length : filteredBuilders.length} results`}</p>
    </>
  );

  return (
    <>
    <div className="mb-8 rounded-xl border bg-muted/30 p-5">
      <p className="text-xs font-semibold uppercase tracking-wider text-primary">Research Toolkit · {promptsUpdatedAt}</p>
      <h2 className="mt-2 text-xl font-semibold">{language === 'ko' ? '연구에서 실제 시스템까지' : 'From research to working systems'}</h2>
      <p className="mt-2 text-sm leading-6 text-muted-foreground">{language === 'ko' ? 'RAG·에이전트·멀티모달·Text-to-SQL의 설계와 평가를 준비하세요. 폼으로 프롬프트를 만들고, 선호하는 AI 도구에서 실행한 결과를 워크플로우로 이어갈 수 있습니다.' : 'Prepare RAG, agents, multimodal systems and Text-to-SQL with design and evaluation templates. Build a prompt, run it in your preferred AI tool, then bring the result into a workflow.'}</p>
      <div className="mt-3 flex flex-wrap gap-2 text-xs">{['RAG', 'Agents', 'Multimodal', 'Text-to-SQL', 'Evals'].map(tag => <button key={tag} type="button" aria-pressed={selectedTags.has(tag)} className={cn('rounded-full border px-3 py-1.5', selectedTags.has(tag) ? 'border-primary bg-primary text-primary-foreground' : 'bg-background hover:bg-accent')} onClick={() => { setTab('builders'); setActiveBuilder(null); setSelectedCats(new Set()); setSelectedTags(new Set([tag])); setQuery(''); }}>{tag}</button>)}</div>
    </div>
    <Tabs value={tab} onValueChange={(v) => { clearPromptTarget(); setTab(v as typeof tab); setActiveBuilder(null); setPendingValues(null); setWorkflowCtx(null); }} className="w-full">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
        <TabsList className="flex-wrap h-auto">
          <TabsTrigger value="builders"><Wand2 className="mr-1.5 h-4 w-4" />{language === 'ko' ? '빌더' : 'Builders'} ({builders.length})</TabsTrigger>
          <TabsTrigger value="library"><Library className="mr-1.5 h-4 w-4" />{language === 'ko' ? '라이브러리' : 'Library'} ({snippets.length})</TabsTrigger>
          <TabsTrigger value="workflows"><Workflow className="mr-1.5 h-4 w-4" />{language === 'ko' ? '워크플로우' : 'Workflows'} ({workflows.length})</TabsTrigger>
          <TabsTrigger value="favorites"><Star className="mr-1.5 h-4 w-4" />{language === 'ko' ? '즐겨찾기' : 'Favorites'} ({favorites.length})</TabsTrigger>
        </TabsList>
      </div>

      {!activeBuilder && (tab === 'builders' || tab === 'library' || tab === 'favorites') && renderToolbar()}
      {workflowSaveFailed && <p role="alert" className="mb-4 text-sm text-red-700 dark:text-red-300">{language === 'ko' ? '워크플로우 진행 기록을 저장하지 못했습니다. 결과를 복사해 보관하고 브라우저 저장 설정을 확인하세요.' : 'Workflow progress could not be saved. Copy your result and check browser storage settings.'}</p>}

      <TabsContent value="builders" className="mt-0">
        {activeBuilder ? (
          <BuilderDetail
            key={`${activeBuilder.id}-${builderVersion}`}
            builder={activeBuilder}
            initialValues={pendingValues}
            workflowCtx={workflowCtx}
            onBack={() => { clearPromptTarget(); setActiveBuilder(null); setPendingValues(null); setWorkflowCtx(null); }}
            isFav={favorites.includes(activeBuilder.id)}
            onToggleFav={() => toggleFav(activeBuilder.id)}
            onShare={handleShare}
            shared={shared}
            shareFailed={shareFailed}
            onWorkflowDone={(output) => {
              if (workflowCtx) {
                if (!saveWorkflowOutput(workflowCtx, output)) { setWorkflowSaveFailed(true); return; }
                setWorkflowSaveFailed(false);
                setActiveBuilder(null);
                setWorkflowCtx(null);
                setPendingValues(null);
                setTab('workflows');
              }
            }}
            language={language}
            L={L}
          />
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filteredBuilders.map((b) => {
              const Icon = ICONS[b.icon] ?? Wand2;
              return (
                <BuilderCard
                  key={b.id}
                  builder={b}
                  Icon={Icon}
                  isFav={favorites.includes(b.id)}
                  onToggleFav={() => toggleFav(b.id)}
                  onOpen={() => openBuilder(b)}
                  catLabel={catLabel(b.category)}
                  catColor={catColor(b.category)}
                  L={L}
                  lang={language}
                />
              );
            })}
            {filteredBuilders.length === 0 && <EmptyState lang={language} />}
          </div>
        )}
      </TabsContent>

      <TabsContent value="library" className="mt-0">
        <div className="flex justify-end mb-3">
          <Button size="sm" variant="outline" onClick={() => { setEditingCustom(null); setShowCustomForm(true); }}>
            <Plus className="mr-1 h-4 w-4" />
            {language === 'ko' ? '내 프롬프트 추가' : 'Add my prompt'}
          </Button>
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredSnippets.map((s) => (
            <SnippetCard
              key={s.id}
              snippet={s}
              isCustom={customIds.has(s.id)}
              expanded={expandedSnip === s.id}
              onToggle={() => setExpandedSnip(expandedSnip === s.id ? null : s.id)}
              isFav={favorites.includes(s.id)}
              onToggleFav={() => toggleFav(s.id)}
              values={snipValues[s.id] ?? {}}
              onValueChange={(name, val) => setSnipValues((p) => ({ ...p, [s.id]: { ...(p[s.id] ?? {}), [name]: val } }))}
              onDelete={customIds.has(s.id) ? () => deleteCustom(s.id) : undefined}
              onEdit={customIds.has(s.id) ? () => { setEditingCustom(customs.find((c) => c.id === s.id) ?? null); setShowCustomForm(true); } : undefined}
              catLabel={catLabel(s.category)}
              catColor={catColor(s.category)}
              L={L}
              lang={language}
            />
          ))}
          {filteredSnippets.length === 0 && <EmptyState lang={language} />}
        </div>
      </TabsContent>

      <TabsContent value="workflows" className="mt-0">
        <div className="grid gap-4 md:grid-cols-2">
          {workflows.map((wf) => {
            const Icon = ICONS[wf.icon] ?? Workflow;
            const state = loadWorkflowState(wf.id);
            return (
              <Card key={wf.id} className="h-full flex flex-col">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary"><Icon className="h-5 w-5" /></div>
                    <div>
                      <CardTitle className="text-base">{L(wf.title)}</CardTitle>
                      <p className="text-sm text-muted-foreground">{L(wf.description)}</p>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col">
                  <ol className="space-y-2 mb-4">
                    {wf.steps.map((step, i) => {
                      const sb = builders.find((b) => b.id === step.builderId);
                      const done = state.outputs[i];
                      return (
                        <li key={i} className={cn('flex items-start gap-2 text-sm rounded-md border p-2', done && 'border-green-500/40 bg-green-500/5')}>
                          <span className={cn('mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold', done ? 'bg-green-700 text-white' : 'bg-muted text-muted-foreground')}>{done ? '✓' : i + 1}</span>
                          <div className="min-w-0">
                            <p className="font-medium">{sb ? L(sb.title) : step.builderId}</p>
                            {step.note && <p className="text-xs text-muted-foreground">{L(step.note)}</p>}
                          </div>
                        </li>
                      );
                    })}
                  </ol>
                  <Button
                    className="mt-auto"
                    size="sm"
                    onClick={() => {
                      const startStep = wf.steps.findIndex((_, i) => !state.outputs[i]);
                      const step = startStep === -1 ? 0 : startStep;
                      const sb = builders.find((b) => b.id === wf.steps[step].builderId);
                      if (sb) {
                        const prevOutput = step > 0 ? state.outputs[step - 1] : '';
                        const prefilled = prevOutput ? prefillBuilder(sb, prevOutput, language) : null;
                        setTab('builders');
                        openBuilder(sb, { values: prefilled, wf: { wf, step } });
                      }
                    }}
                  >
                    <Workflow className="mr-1.5 h-4 w-4" />
                    {language === 'ko' ? '시작 / 이어하기' : 'Start / Resume'}
                  </Button>
                  {state.outputs.some(Boolean) && <Button variant="ghost" size="sm" className="mt-2" onClick={() => {
                    resetWorkflowState(wf.id);
                    setWorkflowRevision(v => v + 1);
                  }}>{language === 'ko' ? '진행 기록 초기화' : 'Reset progress'}</Button>}
                </CardContent>
              </Card>
            );
          })}
        </div>
      </TabsContent>

      <TabsContent value="favorites" className="mt-0">
        {favorites.length === 0 ? (
          <div className="text-center py-16 text-muted-foreground">
            <Star className="h-10 w-10 mx-auto mb-3 opacity-40" />
            <p>{language === 'ko' ? '즐겨찾기한 프롬프트가 없습니다. 카드의 ★를 눌러 추가하세요.' : 'No favorites yet. Tap ★ on a card to add.'}</p>
          </div>
        ) : (
          <>
            {favBuilders.length > 0 && (
              <>
                <p className="text-sm font-medium text-muted-foreground mb-3 mt-2">{language === 'ko' ? '빌더' : 'Builders'}</p>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 mb-8">
                  {favBuilders.map((b) => {
                    const Icon = ICONS[b.icon] ?? Wand2;
                    return (
                      <BuilderCard key={b.id} builder={b} Icon={Icon} isFav onToggleFav={() => toggleFav(b.id)} onOpen={() => openBuilder(b)} catLabel={catLabel(b.category)} catColor={catColor(b.category)} L={L} lang={language} />
                    );
                  })}
                </div>
              </>
            )}
            {favSnippets.length > 0 && (
              <>
                <p className="text-sm font-medium text-muted-foreground mb-3">{language === 'ko' ? '라이브러리' : 'Library'}</p>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {favSnippets.map((s) => (
                    <SnippetCard key={s.id} snippet={s} isCustom={customIds.has(s.id)} expanded={expandedSnip === s.id} onToggle={() => setExpandedSnip(expandedSnip === s.id ? null : s.id)} isFav onToggleFav={() => toggleFav(s.id)} values={snipValues[s.id] ?? {}} onValueChange={(name, val) => setSnipValues((p) => ({ ...p, [s.id]: { ...(p[s.id] ?? {}), [name]: val } }))} catLabel={catLabel(s.category)} catColor={catColor(s.category)} L={L} lang={language} />
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </TabsContent>

      {showCustomForm && (
        <CustomPromptForm
          initial={editingCustom}
          categories={categories}
          onClose={() => { setShowCustomForm(false); setEditingCustom(null); }}
          onSave={(c) => { saveCustom(c); setShowCustomForm(false); setEditingCustom(null); }}
          lang={language}
          L={L}
        />
      )}
    </Tabs>
    <div className="mt-10 border-t pt-5 text-xs text-muted-foreground">
      <p className="font-medium">{language === 'ko' ? '프롬프트 작성·평가 참고 자료' : 'Prompt design and evaluation references'}</p>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-2">{promptReferences.map(ref => <a key={ref.url} href={ref.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 hover:text-primary hover:underline">{ref.title}<ExternalLink aria-hidden="true" className="h-3 w-3" /></a>)}</div>
      <p className="mt-3">{language === 'ko' ? '빌더는 프롬프트를 작성합니다. 모델 실행과 결과 검증은 사용하는 AI 도구에서 진행하세요.' : 'Builders compose prompts. Run models and verify results in your chosen AI tool.'}</p>
    </div>
    </>
  );

}
