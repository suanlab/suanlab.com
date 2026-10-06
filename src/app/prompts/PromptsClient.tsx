'use client';

import { useState, useMemo, useEffect, useCallback, useId } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { promptsUpdatedAt, promptReferences } from '@/data/prompts/advanced';
import { defaultPromptValues, normalizePromptValues, encodePromptShare as encodeShare, decodePromptShare as decodeShare, detectPromptVariables as detectVariables, substitutePromptVariables as substitute } from '@/lib/prompt-state';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  Search, Copy, Check, Download, Star, ArrowLeft, Wand2, Lightbulb,
  ShieldCheck, FileSearch, Compass, FlaskConical, PenLine, Target,
  MessageSquareReply, Library, FileText, Database, Presentation,
  GraduationCap, RotateCcw, ExternalLink, Share2, RefreshCw, Plus,
  Trash2, Edit3, X, Workflow, ChevronRight, Lightbulb as TipIcon,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useLanguage } from '@/components/language-provider';
import {
  promptBuilders,
  promptSnippets,
  promptCategories,
  type PromptBuilder,
  type PromptSnippet,
  type PromptCategory,
  type PromptField,
  type LocalizedText,
  type PromptWorkflow,
} from '@/data/prompts';
import { promptWorkflows } from '@/data/prompts/workflows';

interface PromptsClientProps {
  workflows?: PromptWorkflow[];
}

interface CustomSnippet {
  id: string;
  title: string;
  content: string;
  category: PromptCategory;
  tags: string[];
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
const WF_KEY = 'suanlab-prompt-workflow-state';

// ─── helpers ───
function estimateTokens(text: string): number {
  if (!text) return 0;
  const korean = (text.match(/[가-힣]/g) || []).length;
  const other = text.length - korean;
  return Math.round(korean / 1.8 + other / 4);
}

function asLocalized(s: string): LocalizedText {
  return { ko: s, en: s };
}

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
    const restoreSharedBuilder = () => {
      const decoded = decodeShare(window.location.hash);
      if (decoded) {
        const b = builders.find((x) => x.id === decoded.b);
        if (b) {
          setTab('builders');
          setActiveBuilder(b);
          setPendingValues(normalizePromptValues(b.fields, decoded.v));
          setWorkflowCtx(null);
          setBuilderVersion(v => v + 1);
        }
      }
    };
    restoreSharedBuilder();
    window.addEventListener('hashchange', restoreSharedBuilder);
    return () => window.removeEventListener('hashchange', restoreSharedBuilder);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
    <Tabs value={tab} onValueChange={(v) => { setTab(v as typeof tab); setActiveBuilder(null); setPendingValues(null); setWorkflowCtx(null); }} className="w-full">
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
            onBack={() => { setActiveBuilder(null); setPendingValues(null); setWorkflowCtx(null); }}
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
                    try { const all = JSON.parse(localStorage.getItem(WF_KEY) ?? '{}'); delete all[wf.id]; localStorage.setItem(WF_KEY, JSON.stringify(all)); } catch { /* Storage is optional. */ }
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

  // ── workflow state helpers (closures over component scope) ──
  function loadWorkflowState(wfId: string): { outputs: string[] } {
    try {
      const raw = localStorage.getItem(WF_KEY);
      const all = raw ? JSON.parse(raw) : {};
      const outputs = all?.[wfId]?.outputs;
      return { outputs: Array.isArray(outputs) && outputs.every(value => typeof value === 'string') ? outputs : [] };
    } catch {
      return { outputs: [] };
    }
  }
  function saveWorkflowOutput(ctx: { wf: PromptWorkflow; step: number }, output: string) {
    try {
      const raw = localStorage.getItem(WF_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      const all = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
      const prev = loadWorkflowState(ctx.wf.id).outputs;
      const outputs = prev.slice(0, ctx.step);
      outputs[ctx.step] = output;
      all[ctx.wf.id] = { outputs };
      localStorage.setItem(WF_KEY, JSON.stringify(all));
      return true;
    } catch {
      return false;
    }
  }
}

function prefillBuilder(b: PromptBuilder, prevOutput: string, language: 'ko' | 'en'): Record<string, string | string[]> {
  const v = defaultPromptValues(b.fields, language);
  const target = b.fields.find(f => f.id === 'context') ?? b.fields.find(f => f.type === 'textarea');
  if (target) v[target.id] = prevOutput;
  return v;
}

function customToSnippet(c: CustomSnippet): PromptSnippet {
  return { id: c.id, title: asLocalized(c.title), description: asLocalized(''), category: c.category, tags: c.tags, content: c.content };
}

// ─── Builder card ───
function BuilderCard({
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
            <span key={t} className="inline-flex items-center rounded bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">#{t}</span>
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

// ─── Builder detail ───
function BuilderDetail({
  builder, initialValues, workflowCtx, onBack, isFav, onToggleFav, onShare, shared, shareFailed, onWorkflowDone, language, L,
}: {
  builder: PromptBuilder;
  initialValues: Record<string, string | string[]> | null;
  workflowCtx: { wf: PromptWorkflow; step: number } | null;
  onBack: () => void;
  isFav: boolean;
  onToggleFav: () => void;
  onShare: (id: string, values: Record<string, string | string[]>) => void;
  shared: boolean;
  shareFailed: boolean;
  onWorkflowDone: (output: string) => void;
  language: 'ko' | 'en';
  L: (t: LocalizedText) => string;
}) {
  const draftKey = `suanlab-prompt-draft-${builder.id}`;
  const [draft] = useState(() => {
    try {
      const value = JSON.parse(localStorage.getItem(draftKey) ?? '{}');
      return { values: normalizePromptValues(builder.fields, value?.values), output: typeof value?.output === 'string' ? value.output : '', edited: value?.edited === true };
    } catch { return { values: {}, output: '', edited: false }; }
  });
  const [values, setValues] = useState<Record<string, string | string[]>>(() => ({
    ...defaultPromptValues(builder.fields, language),
    ...(initialValues ? {} : draft.values), ...normalizePromptValues(builder.fields, initialValues),
  }));
  const [output, setOutput] = useState(initialValues ? '' : draft.output);
  const [edited, setEdited] = useState(!initialValues && draft.edited);
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);
  const [missing, setMissing] = useState<Set<string>>(new Set());
  const [showExample, setShowExample] = useState(false);
  const [workflowResult, setWorkflowResult] = useState('');
  const [storageBlocked, setStorageBlocked] = useState(false);

  const prompt = useMemo(() => {
    try {
      return builder.generate(values);
    } catch {
      return '';
    }
  }, [builder, values]);

  useEffect(() => { if (!edited) setOutput(prompt); }, [prompt, edited]);
  useEffect(() => {
    try { localStorage.setItem(draftKey, JSON.stringify({ values, output, edited })); }
    catch { setStorageBlocked(true); }
  }, [draftKey, values, output, edited]);

  const setValue = (id: string, val: string | string[]) => {
    setValues((prev) => ({ ...prev, [id]: val }));
    setMissing((prev) => {
      if (!prev.has(id)) return prev;
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  };

  const validate = () => {
    const reqMissing = builder.fields.filter((f) => f.required && f.type !== 'multiselect' && !String(values[f.id] ?? '').trim());
    const mulMissing = builder.fields.filter((f) => f.required && f.type === 'multiselect' && (values[f.id] as string[]).length === 0);
    if ([...reqMissing, ...mulMissing].length > 0) {
      setMissing(new Set([...reqMissing, ...mulMissing].map((f) => f.id)));
      return false;
    }
    return true;
  };
  const handleCopy = async () => {
    if (!validate() || !output.trim()) return;
    setCopyFailed(false);
    try {
      await navigator.clipboard.writeText(output);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopyFailed(true);
    }
  };

  const handleDownload = () => {
    if (!validate() || !output.trim()) return;
    const blob = new Blob([`# ${L(builder.title)}\n\n${output}\n`], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${builder.id}.md`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const tokens = estimateTokens(output);

  return (
    <div>
      <div className="flex items-center justify-between gap-2 mb-4">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="mr-1.5 h-4 w-4" />
          {language === 'ko' ? '목록으로' : 'Back'}
        </Button>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={() => {
            setValues(defaultPromptValues(builder.fields, language));
            setEdited(false); setMissing(new Set()); setWorkflowResult('');
          }}><RotateCcw className="mr-1.5 h-4 w-4" />{language === 'ko' ? '초안 초기화' : 'Clear draft'}</Button>
          <Button variant="ghost" size="sm" onClick={onToggleFav}>
            <Star className={cn('mr-1.5 h-4 w-4', isFav && 'fill-amber-400 text-amber-400')} />
            {isFav ? (language === 'ko' ? '즐겨찾기됨' : 'Favorited') : (language === 'ko' ? '즐겨찾기' : 'Favorite')}
          </Button>
        </div>
      </div>

      {workflowCtx && (
        <div className="mb-4 rounded-md border border-primary/30 bg-primary/5 p-3 text-sm">
          <p className="font-medium">{language === 'ko' ? `워크플로우: ${L(workflowCtx.wf.title)}` : `Workflow: ${L(workflowCtx.wf.title)}`}</p>
          <p className="text-xs text-muted-foreground">{language === 'ko' ? `단계 ${workflowCtx.step + 1} / ${workflowCtx.wf.steps.length}` : `Step ${workflowCtx.step + 1} / ${workflowCtx.wf.steps.length}`}</p>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Form */}
        <div>
          <h2 className="text-xl font-bold mb-1">{L(builder.title)}</h2>
          <p className="text-sm text-muted-foreground mb-4">{L(builder.description)}</p>
          {builder.tips && builder.tips.length > 0 && (
            <div className="mb-4 rounded-md border border-amber-500/30 bg-amber-500/5 p-3">
              <p className="text-xs font-semibold text-amber-700 dark:text-amber-400 mb-1 flex items-center gap-1"><TipIcon className="h-3.5 w-3.5" />{language === 'ko' ? '사용 팁' : 'Tips'}</p>
              <ul className="text-xs text-muted-foreground space-y-1 list-disc pl-4">
                {builder.tips.map((t, i) => <li key={i}>{L(t)}</li>)}
              </ul>
            </div>
          )}
          <div className="space-y-4">
            {builder.fields.map((f) => (
              <FieldInput key={f.id} field={f} value={values[f.id]} onChange={(val) => setValue(f.id, val)} invalid={missing.has(f.id)} L={L} />
            ))}
          </div>
          {builder.example && (
            <div className="mt-4">
              <button onClick={() => setShowExample((s) => !s)} className="text-xs text-primary hover:underline flex items-center gap-1">
                <ChevronRight className={cn('h-3 w-3 transition-transform', showExample && 'rotate-90')} />
                {language === 'ko' ? '이런 결과가 나와요 (예시)' : 'Example output'}
              </button>
              {showExample && <pre className="mt-2 whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-3 text-xs font-mono max-h-60 overflow-y-auto">{builder.example}</pre>}
            </div>
          )}
        </div>

        {/* Output */}
        <div className="lg:sticky lg:top-24 self-start">
          <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
            <span className="text-sm font-medium text-muted-foreground">
              {language === 'ko' ? '생성된 프롬프트' : 'Generated prompt'}
              <span className="ml-2 text-[11px] text-muted-foreground">≈ {tokens} tokens</span>
            </span>
            <div className="flex gap-2">
              <Button size="sm" variant="ghost" onClick={() => { setOutput(prompt); setEdited(false); }} aria-label={language === 'ko' ? '현재 폼으로 프롬프트 업데이트' : 'Update prompt from current form'} title={language === 'ko' ? '재생성' : 'Regenerate'}>
                <RefreshCw className="h-3.5 w-3.5" />
              </Button>
              <Button size="sm" variant="ghost" onClick={() => onShare(builder.id, values)} aria-label={language === 'ko' ? '입력값을 포함한 공유 링크 복사' : 'Copy share link containing inputs'} title={language === 'ko' ? '공유 URL 복사' : 'Copy share URL'}>
                {shared ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Share2 className="h-3.5 w-3.5" />}
              </Button>
              <Button size="sm" variant="outline" onClick={handleDownload} disabled={!output}><Download className="mr-1 h-3.5 w-3.5" />.md</Button>
              <Button size="sm" onClick={handleCopy}>
                {copied ? <Check className="mr-1 h-3.5 w-3.5 text-green-500" /> : <Copy className="mr-1 h-3.5 w-3.5" />}
                {copied ? (language === 'ko' ? '복사됨!' : 'Copied!') : (language === 'ko' ? '복사' : 'Copy')}
              </Button>
            </div>
          </div>
          {missing.size > 0 && <p role="alert" className="text-xs text-red-700 dark:text-red-300 mb-2">{language === 'ko' ? '필수 항목을 입력해 주세요.' : 'Please fill required fields.'}</p>}
          <textarea
            value={output}
            aria-label={language === 'ko' ? '생성된 프롬프트 편집' : 'Edit generated prompt'}
            onChange={(e) => { setOutput(e.target.value); setEdited(true); }}
            spellCheck={false}
            className="w-full min-h-[280px] max-h-[60vh] overflow-y-auto whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-4 text-xs font-mono leading-relaxed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            placeholder={language === 'ko' ? '왼쪽 폼을 채우면 프롬프트가 생성됩니다. 직접 수정도 가능합니다.' : 'Fill the form to generate. You can also edit directly.'}
          />
          <p className="mt-2 text-[11px] text-muted-foreground flex items-center gap-1">
            <ExternalLink className="h-3 w-3" />
            {language === 'ko' ? (edited ? '직접 수정한 내용을 유지합니다. 폼 변경을 반영하려면 업데이트 버튼을 누르세요.' : '폼을 채우면 자동 생성됩니다. 초안은 이 브라우저에 저장됩니다.') : (edited ? 'Your edits are preserved. Use Update to apply form changes.' : 'The form generates a prompt. Drafts are saved in this browser.')}
          </p>
          {(copyFailed || shareFailed) && <p role="alert" className="mt-2 text-xs text-red-700 dark:text-red-300">{language === 'ko' ? '복사하지 못했습니다. 프롬프트를 선택하여 직접 복사하세요.' : 'Copy failed. Select the prompt and copy it manually.'}</p>}
          {storageBlocked && <p role="status" className="mt-2 text-xs text-muted-foreground">{language === 'ko' ? '이 브라우저에서 초안을 저장하지 못했습니다.' : 'Draft storage is unavailable in this browser.'}</p>}
          <p className="mt-2 text-xs text-muted-foreground">{language === 'ko' ? '공유 링크에는 폼 입력값이 포함됩니다.' : 'Share links include the form inputs.'}</p>
          {workflowCtx && <div className="mt-5 rounded-lg border p-3">
            <label htmlFor="workflow-result" className="text-sm font-medium">{language === 'ko' ? 'AI 도구에서 실행한 결과' : 'Result from your AI tool'}</label>
            <textarea id="workflow-result" value={workflowResult} onChange={e => setWorkflowResult(e.target.value)} className="mt-2 min-h-28 w-full rounded-md border bg-background p-3 text-sm" />
            <p className="mt-2 text-xs text-muted-foreground">{language === 'ko' ? '실제 응답을 붙여넣으면 다음 단계의 입력으로 이어집니다.' : 'Paste the actual response to carry it into the next step.'}</p>
            <Button className="mt-3 w-full" size="sm" disabled={!workflowResult.trim()} onClick={() => { if (validate()) onWorkflowDone(workflowResult.trim()); }}>{language === 'ko' ? '결과 저장 & 단계 완료' : 'Save result & complete step'}</Button>
          </div>}
        </div>
      </div>
    </div>
  );
}

// ─── Field input ───
function FieldInput({
  field, value, onChange, invalid, L,
}: {
  field: PromptField;
  value: string | string[];
  onChange: (v: string | string[]) => void;
  invalid: boolean;
  L: (t: LocalizedText) => string;
}) {
  const fieldId = useId();
  const label = (
    <label htmlFor={field.type === 'multiselect' ? undefined : fieldId} id={`${fieldId}-label`} className="text-sm font-medium flex items-center gap-1">
      {L(field.label)}
      {field.required && <span aria-hidden="true" className="text-red-700 dark:text-red-300">*</span>}
    </label>
  );
  const help = field.help && <p id={`${fieldId}-help`} className="text-xs text-muted-foreground mt-0.5">{L(field.help)}</p>;

  if (field.type === 'textarea') {
    return (
      <div>
        {label}
        <textarea
          id={fieldId} aria-required={field.required} aria-invalid={invalid} aria-describedby={field.help ? `${fieldId}-help` : undefined}
          value={value as string}
          onChange={(e) => onChange(e.target.value)}
          placeholder={field.placeholder ? L(field.placeholder) : undefined}
          className={cn('mt-1 flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2', invalid && 'border-red-500')}
        />
        {help}
      </div>
    );
  }
  if (field.type === 'select') {
    return (
      <div>
        {label}
        <select id={fieldId} aria-required={field.required} aria-invalid={invalid} aria-describedby={field.help ? `${fieldId}-help` : undefined} value={value as string} onChange={(e) => onChange(e.target.value)} className={cn('mt-1 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2', invalid && 'border-red-500')}>
          {field.options?.map((o) => <option key={o.value} value={o.value}>{L(o.label)}</option>)}
        </select>
        {help}
      </div>
    );
  }
  if (field.type === 'multiselect') {
    const arr = value as string[];
    const toggle = (val: string) => onChange(arr.includes(val) ? arr.filter((x) => x !== val) : [...arr, val]);
    return (
      <div>
        {label}
        <div role="group" aria-labelledby={`${fieldId}-label`} aria-describedby={field.help ? `${fieldId}-help` : undefined} className="mt-1.5 flex flex-wrap gap-1.5">
          {field.options?.map((o) => {
            const on = arr.includes(o.value);
            const lbl = L(o.label);
            return (
              <button key={o.value} type="button" onClick={() => toggle(o.value)} aria-pressed={on} className={cn('rounded-full border px-2.5 py-1 text-xs transition-colors', on ? 'border-primary bg-primary text-primary-foreground' : 'border-input bg-background hover:bg-accent')}>
                {lbl.length > 42 ? lbl.slice(0, 40) + '…' : lbl}
              </button>
            );
          })}
        </div>
        {help}
      </div>
    );
  }
  return (
    <div>
      {label}
      <Input id={fieldId} aria-required={field.required} aria-invalid={invalid} aria-describedby={field.help ? `${fieldId}-help` : undefined} value={value as string} onChange={(e) => onChange(e.target.value)} placeholder={field.placeholder ? L(field.placeholder) : undefined} className={cn('mt-1', invalid && 'border-red-500')} />
      {help}
    </div>
  );
}

// ─── Snippet card ───
function SnippetCard({
  snippet, isCustom, expanded, onToggle, isFav, onToggleFav, values, onValueChange, onDelete, onEdit, catLabel, catColor, L, lang,
}: {
  snippet: PromptSnippet;
  isCustom: boolean;
  expanded: boolean;
  onToggle: () => void;
  isFav: boolean;
  onToggleFav: () => void;
  values: Record<string, string>;
  onValueChange: (name: string, val: string) => void;
  onDelete?: () => void;
  onEdit?: () => void;
  catLabel: string;
  catColor: string;
  L: (t: LocalizedText) => string;
  lang: 'ko' | 'en';
}) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const contentId = useId();
  const content = lang === 'en' && snippet.contentEn ? snippet.contentEn : snippet.content;
  const declared = detectVariables(content);
  const defaults = Object.fromEntries((snippet.variables ?? []).map(v => [v.name, v.default ?? '']));
  const filledValues = { ...defaults, ...values };
  const finalContent = substitute(content, filledValues);
  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setCopyError(false);
    if (declared.some(name => !filledValues[name]?.trim())) { setCopyError(true); if (!expanded) onToggle(); return; }
    try {
      await navigator.clipboard.writeText(finalContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopyError(true);
    }
  };
  return (
    <Card className="h-full flex flex-col transition-all hover:shadow-md">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base hover:text-primary transition-colors flex items-center gap-1.5">
            <button type="button" onClick={onToggle} aria-expanded={expanded} aria-controls={expanded ? contentId : undefined} className="text-left">{L(snippet.title)}</button>
            {isCustom && <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[9px] font-bold text-primary">MINE</span>}
          </CardTitle>
          <div className="flex items-center gap-1 shrink-0">
            {isCustom && onEdit && <button onClick={(e) => { e.stopPropagation(); onEdit(); }} aria-label="edit" className="text-muted-foreground hover:text-primary"><Edit3 className="h-4 w-4" /></button>}
            {isCustom && onDelete && <button onClick={(e) => { e.stopPropagation(); onDelete(); }} aria-label="delete" className="text-muted-foreground hover:text-red-500"><Trash2 className="h-4 w-4" /></button>}
            <button onClick={(e) => { e.stopPropagation(); onToggleFav(); }} aria-pressed={isFav} aria-label={lang === 'ko' ? '즐겨찾기 변경' : 'Toggle favorite'} className="text-muted-foreground hover:text-amber-500"><Star className={cn('h-5 w-5', isFav && 'fill-amber-400 text-amber-400')} /></button>
          </div>
        </div>
        <p className="text-sm text-muted-foreground line-clamp-2">{L(snippet.description)}</p>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col">
        <div className="flex flex-wrap gap-1.5 mb-3">
          <span className={cn('inline-flex items-center rounded px-2 py-0.5 text-[10px] font-medium', catColor)}>{catLabel}</span>
          {snippet.tags.slice(0, 2).map((t) => <span key={t} className="inline-flex items-center rounded bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">#{t}</span>)}
          {declared.length > 0 && <span className="inline-flex items-center rounded bg-blue-100 px-2 py-0.5 text-[10px] text-blue-800 dark:bg-blue-900/40 dark:text-blue-300">{declared.length} {lang === 'ko' ? '변수' : 'vars'}</span>}
        </div>
        {expanded && declared.length > 0 && (
          <div className="mb-3 space-y-2 rounded-md border p-2">
            {declared.map((name) => {
              const meta = snippet.variables?.find((vv) => vv.name === name);
              return (
                <div key={name}>
                  <label htmlFor={`${contentId}-${name}`} className="text-xs font-medium">{meta ? L(meta.label) : name}</label>
                  <Input id={`${contentId}-${name}`} value={values[name] ?? meta?.default ?? ''} onChange={(e) => onValueChange(name, e.target.value)} placeholder={name} className="mt-0.5 h-8 text-xs" />
                </div>
              );
            })}
          </div>
        )}
        {expanded && (
          <pre id={contentId} className="whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-3 text-xs font-mono leading-relaxed max-h-[40vh] overflow-y-auto mb-3">{finalContent}</pre>
        )}
        {copyError && <p role="alert" className="mb-2 text-xs text-red-700 dark:text-red-300">{lang === 'ko' ? '변수를 모두 채워 주세요. 복사가 실패하면 본문을 직접 선택해 복사하세요.' : 'Fill all variables. If copying fails, select and copy the text manually.'}</p>}
        <div className="mt-auto flex gap-2">
          <Button className="flex-1" size="sm" variant={expanded ? 'outline' : 'default'} onClick={onToggle}>{expanded ? (lang === 'ko' ? '접기' : 'Collapse') : (lang === 'ko' ? '보기' : 'View')}</Button>
          <Button size="sm" variant="outline" onClick={handleCopy}>{copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}<span className="ml-1">{copied ? (lang === 'ko' ? '복사됨' : 'Copied') : (lang === 'ko' ? '복사' : 'Copy')}</span></Button>
        </div>
      </CardContent>
    </Card>
  );
}

// ─── Custom prompt form ───
function CustomPromptForm({
  initial, categories, onClose, onSave, lang, L,
}: {
  initial: CustomSnippet | null;
  categories: { id: PromptCategory; label: LocalizedText }[];
  onClose: () => void;
  onSave: (c: CustomSnippet) => void;
  lang: 'ko' | 'en';
  L: (t: LocalizedText) => string;
}) {
  const [title, setTitle] = useState(initial?.title ?? '');
  const [content, setContent] = useState(initial?.content ?? '');
  const [category, setCategory] = useState<PromptCategory>(initial?.category ?? 'coding');
  const [tags, setTags] = useState((initial?.tags ?? []).join(', '));
  const [returnFocus] = useState(() => document.activeElement instanceof HTMLElement ? document.activeElement : null);
  const formId = useId();

  const submit = () => {
    if (!title.trim() || !content.trim()) return;
    onSave({
      id: initial?.id ?? `custom-${Date.now()}`,
      title: title.trim(),
      content: content.trim(),
      category,
      tags: tags.split(',').map((t) => t.trim()).filter(Boolean),
    });
  };

  return (
    <Dialog.Root open onOpenChange={open => { if (!open) onClose(); }}><Dialog.Portal>
      <Dialog.Overlay className="fixed inset-0 z-50 bg-black/50" />
      <Dialog.Content onCloseAutoFocus={event => { event.preventDefault(); if (returnFocus?.isConnected) returnFocus.focus(); }} className="fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[calc(100%_-_2rem)] max-w-lg -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border bg-background p-5">
        <Dialog.Description className="sr-only">{lang === 'ko' ? '개인 프롬프트를 이 브라우저에 저장합니다.' : 'Save a personal prompt in this browser.'}</Dialog.Description>
        <div className="flex items-center justify-between mb-4">
          <Dialog.Title className="font-bold">{lang === 'ko' ? (initial ? '내 프롬프트 수정' : '내 프롬프트 추가') : (initial ? 'Edit my prompt' : 'Add my prompt')}</Dialog.Title>
          <button onClick={onClose} aria-label={lang === 'ko' ? '닫기' : 'Close'}><X className="h-5 w-5 text-muted-foreground" /></button>
        </div>
        <div className="space-y-3">
          <div>
            <label htmlFor={`${formId}-title`} className="text-sm font-medium">{lang === 'ko' ? '제목' : 'Title'}<span aria-hidden="true" className="text-red-700 dark:text-red-300">*</span></label>
            <Input id={`${formId}-title`} aria-required value={title} onChange={(e) => setTitle(e.target.value)} className="mt-1" />
          </div>
          <div>
            <label htmlFor={`${formId}-content`} className="text-sm font-medium">{lang === 'ko' ? '본문' : 'Content'}<span aria-hidden="true" className="text-red-700 dark:text-red-300">*</span></label>
            <textarea id={`${formId}-content`} aria-required value={content} onChange={(e) => setContent(e.target.value)} className="mt-1 flex min-h-[160px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" />
            <p className="text-xs text-muted-foreground mt-1">{lang === 'ko' ? '{{변수}} 문법으로 변수를 넣으면 확장 시 채울 수 있습니다.' : 'Use {{variable}} placeholders to fill in later.'}</p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label htmlFor={`${formId}-category`} className="text-sm font-medium">{lang === 'ko' ? '분야' : 'Category'}</label>
              <select id={`${formId}-category`} value={category} onChange={(e) => setCategory(e.target.value as PromptCategory)} className="mt-1 flex h-10 w-full rounded-md border border-input bg-background px-3 text-sm">
                {categories.map((c) => <option key={c.id} value={c.id}>{L(c.label)}</option>)}
              </select>
            </div>
            <div>
              <label htmlFor={`${formId}-tags`} className="text-sm font-medium">{lang === 'ko' ? '태그 (쉼표)' : 'Tags (comma)'}</label>
              <Input id={`${formId}-tags`} value={tags} onChange={(e) => setTags(e.target.value)} className="mt-1" />
            </div>
          </div>
        </div>
        <div className="flex justify-end gap-2 mt-5">
          <Button variant="outline" size="sm" onClick={onClose}>{lang === 'ko' ? '취소' : 'Cancel'}</Button>
          <Button size="sm" onClick={submit} disabled={!title.trim() || !content.trim()}>{lang === 'ko' ? '저장' : 'Save'}</Button>
        </div>
      </Dialog.Content>
    </Dialog.Portal></Dialog.Root>
  );
}

function EmptyState({ lang }: { lang: 'ko' | 'en' }) {
  return (
    <div className="text-center py-16 text-muted-foreground col-span-full">
      <Search className="h-10 w-10 mx-auto mb-3 opacity-40" />
      <p>{lang === 'ko' ? '조건에 맞는 프롬프트가 없습니다.' : 'No prompts match your search.'}</p>
    </div>
  );
}
