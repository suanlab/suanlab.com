'use client';

import { useState, useMemo, useEffect, useCallback } from 'react';
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

function detectVariables(content: string): string[] {
  const set = new Set<string>();
  const re = /\{\{(\w+)\}\}/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(content))) set.add(m[1]);
  return [...set];
}

function substitute(content: string, values: Record<string, string>): string {
  return content.replace(/\{\{(\w+)\}\}/g, (_, n) => values[n]?.trim() || `{{${n}}}`);
}

function encodeShare(builderId: string, values: Record<string, string | string[]>): string {
  try {
    const json = JSON.stringify({ b: builderId, v: values });
    return btoa(encodeURIComponent(json));
  } catch {
    return '';
  }
}

function decodeShare(hash: string): { b: string; v: Record<string, string | string[]> } | null {
  try {
    const json = decodeURIComponent(atob(hash));
    const parsed = JSON.parse(json);
    if (parsed && typeof parsed.b === 'string' && parsed.v) return parsed;
    return null;
  } catch {
    return null;
  }
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
  const L = (t: LocalizedText) => t[language];

  const [tab, setTab] = useState<'builders' | 'library' | 'workflows' | 'favorites'>('builders');
  const [query, setQuery] = useState('');
  const [selectedCats, setSelectedCats] = useState<Set<PromptCategory>>(new Set());
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());
  const [sort, setSort] = useState<'default' | 'alpha' | 'recent'>('default');

  const [activeBuilder, setActiveBuilder] = useState<PromptBuilder | null>(null);
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
      if (fav) setFavorites(JSON.parse(fav));
      const cus = localStorage.getItem(CUSTOM_KEY);
      if (cus) setCustoms(JSON.parse(cus));
    } catch {
      /* noop */
    }
    if (typeof window !== 'undefined' && window.location.hash.length > 1) {
      const decoded = decodeShare(window.location.hash.slice(1));
      if (decoded) {
        const b = builders.find((x) => x.id === decoded.b);
        if (b) {
          setActiveBuilder(b);
          setPendingValues(decoded.v);
          setTab('builders');
        }
      }
    }
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
    return [...set].sort((a, b) => a.localeCompare(b));
  }, [builders, snippets]);

  const q = query.trim().toLowerCase();
  const matchText = (hay: string) => !q || hay.toLowerCase().includes(q);
  const catOk = (c: PromptCategory) => selectedCats.size === 0 || selectedCats.has(c);
  const tagOk = (tags: string[]) => selectedTags.size === 0 || tags.some((t) => selectedTags.has(t));

  const sortFn = useCallback(
    <T extends { title: LocalizedText | string }>(arr: T[]): T[] => {
      if (sort === 'alpha') return [...arr].sort((a, b) => L(a.title as LocalizedText).localeCompare(L(b.title as LocalizedText)));
      return arr;
    },
    [sort, language],
  );

  const filteredBuilders = useMemo(
    () =>
      sortFn(
        builders.filter(
          (b) => catOk(b.category) && tagOk(b.tags) && matchText(`${L(b.title)} ${L(b.description)} ${b.tags.join(' ')}`),
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
          return catOk(s.category) && tagOk(s.tags) && matchText(`${title} ${desc} ${s.tags.join(' ')} ${s.content}`);
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
        navigator.clipboard?.writeText(url).then(
          () => {
            setShared(true);
            setTimeout(() => setShared(false), 2000);
          },
          () => {},
        );
      }
    },
    [],
  );

  const openBuilder = (b: PromptBuilder, opts?: { values?: Record<string, string | string[]> | null; wf?: typeof workflowCtx }) => {
    setActiveBuilder(b);
    setPendingValues(opts?.values ?? null);
    setWorkflowCtx(opts?.wf ?? null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const renderToolbar = () => (
    <>
      <div className="relative mb-3">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder={language === 'ko' ? '프롬프트 검색...' : 'Search prompts...'}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="pl-9"
        />
      </div>
      <div className="flex flex-wrap gap-2 mb-3">
        {categories.map((c) => {
          const active = selectedCats.has(c.id);
          return (
            <Button key={c.id} variant={active ? 'default' : 'outline'} size="sm" onClick={() => toggleCat(c.id)} className={cn(!active && 'hover:bg-accent')}>
              {L(c.label)}
            </Button>
          );
        })}
        {(selectedCats.size > 0 || selectedTags.size > 0) && (
          <Button variant="ghost" size="sm" onClick={() => { setSelectedCats(new Set()); setSelectedTags(new Set()); }}>
            <RotateCcw className="mr-1 h-3 w-3" />
            {language === 'ko' ? '초기화' : 'Reset'}
          </Button>
        )}
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as typeof sort)}
          className="ml-auto h-9 rounded-md border border-input bg-background px-2 text-xs"
        >
          <option value="default">{language === 'ko' ? '기본 순' : 'Default'}</option>
          <option value="alpha">{language === 'ko' ? '가나다' : 'A→Z'}</option>
        </select>
      </div>
      {allTags.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-4">
          {allTags.slice(0, 16).map((t) => {
            const on = selectedTags.has(t);
            return (
              <button key={t} type="button" onClick={() => toggleTag(t)} className={cn('rounded-full border px-2 py-0.5 text-[11px] transition-colors', on ? 'border-primary bg-primary text-primary-foreground' : 'border-input bg-background hover:bg-accent text-muted-foreground')}>
                #{t}
              </button>
            );
          })}
        </div>
      )}
    </>
  );

  return (
    <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)} className="w-full">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
        <TabsList className="flex-wrap h-auto">
          <TabsTrigger value="builders"><Wand2 className="mr-1.5 h-4 w-4" />{language === 'ko' ? '빌더' : 'Builders'} ({builders.length})</TabsTrigger>
          <TabsTrigger value="library"><Library className="mr-1.5 h-4 w-4" />{language === 'ko' ? '라이브러리' : 'Library'} ({snippets.length})</TabsTrigger>
          <TabsTrigger value="workflows"><Workflow className="mr-1.5 h-4 w-4" />{language === 'ko' ? '워크플로우' : 'Workflows'} ({workflows.length})</TabsTrigger>
          <TabsTrigger value="favorites"><Star className="mr-1.5 h-4 w-4" />{language === 'ko' ? '즐겨찾기' : 'Favorites'} ({favorites.length})</TabsTrigger>
        </TabsList>
      </div>

      {!activeBuilder && (tab === 'builders' || tab === 'library' || tab === 'favorites') && renderToolbar()}

      <TabsContent value="builders" className="mt-0">
        {activeBuilder ? (
          <BuilderDetail
            builder={activeBuilder}
            initialValues={pendingValues}
            workflowCtx={workflowCtx}
            onBack={() => { setActiveBuilder(null); setPendingValues(null); setWorkflowCtx(null); }}
            isFav={favorites.includes(activeBuilder.id)}
            onToggleFav={() => toggleFav(activeBuilder.id)}
            onShare={handleShare}
            shared={shared}
            onWorkflowDone={(output) => {
              if (workflowCtx) {
                saveWorkflowOutput(workflowCtx, output);
                const nextStep = workflowCtx.step + 1;
                const wf = workflowCtx.wf;
                setActiveBuilder(null);
                setWorkflowCtx(null);
                setPendingValues(null);
                if (nextStep < wf.steps.length) {
                  setTab('workflows');
                }
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
                          <span className={cn('mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold', done ? 'bg-green-500 text-white' : 'bg-muted text-muted-foreground')}>{done ? '✓' : i + 1}</span>
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
                      const startStep = state.outputs.findIndex((o) => !o);
                      const step = startStep === -1 ? 0 : startStep;
                      const sb = builders.find((b) => b.id === wf.steps[step].builderId);
                      if (sb) {
                        const prevOutput = step > 0 ? state.outputs[step - 1] : '';
                        const prefilled = prevOutput ? prefillBuilder(sb, prevOutput) : null;
                        setTab('builders');
                        openBuilder(sb, { values: prefilled, wf: { wf, step } });
                      }
                    }}
                  >
                    <Workflow className="mr-1.5 h-4 w-4" />
                    {language === 'ko' ? '시작 / 이어하기' : 'Start / Resume'}
                  </Button>
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
  );

  // ── workflow state helpers (closures over component scope) ──
  function loadWorkflowState(wfId: string): { outputs: string[] } {
    try {
      const raw = localStorage.getItem(WF_KEY);
      const all = raw ? JSON.parse(raw) : {};
      return all[wfId] ?? { outputs: [] };
    } catch {
      return { outputs: [] };
    }
  }
  function saveWorkflowOutput(ctx: { wf: PromptWorkflow; step: number }, output: string) {
    try {
      const raw = localStorage.getItem(WF_KEY);
      const all = raw ? JSON.parse(raw) : {};
      const prev = all[ctx.wf.id]?.outputs ?? [];
      const outputs = [...prev];
      outputs[ctx.step] = output;
      all[ctx.wf.id] = { outputs };
      localStorage.setItem(WF_KEY, JSON.stringify(all));
    } catch {
      /* noop */
    }
  }
}

function prefillBuilder(b: PromptBuilder, prevOutput: string): Record<string, string | string[]> {
  const v: Record<string, string | string[]> = {};
  let filled = false;
  for (const f of b.fields) {
    if (f.type === 'multiselect') v[f.id] = [];
    else if (!filled && (f.type === 'textarea' || f.type === 'text')) {
      v[f.id] = prevOutput;
      filled = true;
    } else if (f.default) v[f.id] = f.default;
    else v[f.id] = '';
  }
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
          <button onClick={(e) => { e.stopPropagation(); onToggleFav(); }} aria-label="favorite" className="text-muted-foreground hover:text-amber-500 transition-colors">
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
  builder, initialValues, workflowCtx, onBack, isFav, onToggleFav, onShare, shared, onWorkflowDone, language, L,
}: {
  builder: PromptBuilder;
  initialValues: Record<string, string | string[]> | null;
  workflowCtx: { wf: PromptWorkflow; step: number } | null;
  onBack: () => void;
  isFav: boolean;
  onToggleFav: () => void;
  onShare: (id: string, values: Record<string, string | string[]>) => void;
  shared: boolean;
  onWorkflowDone: (output: string) => void;
  language: 'ko' | 'en';
  L: (t: LocalizedText) => string;
}) {
  const buildInitial = useCallback((): Record<string, string | string[]> => {
    const base: Record<string, string | string[]> = {};
    for (const f of builder.fields) {
      if (f.type === 'multiselect') base[f.id] = [];
      else if (f.default) base[f.id] = f.default;
      else base[f.id] = '';
    }
    if (initialValues) {
      for (const k of Object.keys(initialValues)) base[k] = initialValues[k];
    }
    return base;
  }, [builder, initialValues]);

  const [values, setValues] = useState<Record<string, string | string[]>>(buildInitial);
  const [output, setOutput] = useState('');
  const [copied, setCopied] = useState(false);
  const [missing, setMissing] = useState<Set<string>>(new Set());
  const [showExample, setShowExample] = useState(false);

  useEffect(() => {
    const v = buildInitial();
    setValues(v);
    setMissing(new Set());
  }, [builder.id, buildInitial]);

  const prompt = useMemo(() => {
    try {
      return builder.generate(values);
    } catch {
      return '';
    }
  }, [builder, values]);

  // live-regenerate output when the form changes (clobbers manual edits, as intended)
  useEffect(() => {
    setOutput(prompt);
  }, [prompt]);

  const setValue = (id: string, val: string | string[]) => {
    setValues((prev) => ({ ...prev, [id]: val }));
    setMissing((prev) => {
      if (!prev.has(id)) return prev;
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  };

  const handleCopy = async () => {
    const reqMissing = builder.fields.filter((f) => f.required && f.type !== 'multiselect' && !String(values[f.id] ?? '').trim());
    const mulMissing = builder.fields.filter((f) => f.required && f.type === 'multiselect' && (values[f.id] as string[]).length === 0);
    if ([...reqMissing, ...mulMissing].length > 0) {
      setMissing(new Set([...reqMissing, ...mulMissing].map((f) => f.id)));
      return;
    }
    try {
      await navigator.clipboard.writeText(output);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* noop */
    }
  };

  const handleDownload = () => {
    const blob = new Blob([`# ${L(builder.title)}\n\n${output}\n`], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${builder.id}.md`;
    a.click();
    URL.revokeObjectURL(url);
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
              <span className="ml-2 text-[11px] text-muted-foreground/70">≈ {tokens} tokens</span>
            </span>
            <div className="flex gap-2">
              <Button size="sm" variant="ghost" onClick={() => setOutput(prompt)} title={language === 'ko' ? '재생성' : 'Regenerate'}>
                <RefreshCw className="h-3.5 w-3.5" />
              </Button>
              <Button size="sm" variant="ghost" onClick={() => onShare(builder.id, values)} title={language === 'ko' ? '공유 URL 복사' : 'Copy share URL'}>
                {shared ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Share2 className="h-3.5 w-3.5" />}
              </Button>
              <Button size="sm" variant="outline" onClick={handleDownload} disabled={!output}><Download className="mr-1 h-3.5 w-3.5" />.md</Button>
              <Button size="sm" onClick={handleCopy}>
                {copied ? <Check className="mr-1 h-3.5 w-3.5 text-green-500" /> : <Copy className="mr-1 h-3.5 w-3.5" />}
                {copied ? (language === 'ko' ? '복사됨!' : 'Copied!') : (language === 'ko' ? '복사' : 'Copy')}
              </Button>
            </div>
          </div>
          {missing.size > 0 && <p className="text-xs text-red-500 mb-2">{language === 'ko' ? '필수 항목을 입력해 주세요.' : 'Please fill required fields.'}</p>}
          <textarea
            value={output}
            onChange={(e) => setOutput(e.target.value)}
            spellCheck={false}
            className="w-full min-h-[280px] max-h-[60vh] overflow-y-auto whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-4 text-xs font-mono leading-relaxed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            placeholder={language === 'ko' ? '왼쪽 폼을 채우면 프롬프트가 생성됩니다. 직접 수정도 가능합니다.' : 'Fill the form to generate. You can also edit directly.'}
          />
          <p className="mt-2 text-[11px] text-muted-foreground flex items-center gap-1">
            <ExternalLink className="h-3 w-3" />
            {language === 'ko' ? '폼을 바꾸면 자동 재생성. 직접 수정 후 복사하세요.' : 'Edits to the form regenerate; edit the box freely before copying.'}
          </p>
          {workflowCtx && (
            <Button className="w-full mt-3" size="sm" onClick={() => onWorkflowDone(output)}>
              {language === 'ko' ? '이 단계 완료' : 'Complete this step'}
            </Button>
          )}
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
  const label = (
    <label className="text-sm font-medium flex items-center gap-1">
      {L(field.label)}
      {field.required && <span className="text-red-500">*</span>}
    </label>
  );
  const help = field.help && <p className="text-xs text-muted-foreground mt-0.5">{L(field.help)}</p>;

  if (field.type === 'textarea') {
    return (
      <div>
        {label}
        <textarea
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
        <select value={value as string} onChange={(e) => onChange(e.target.value)} className={cn('mt-1 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2', invalid && 'border-red-500')}>
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
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {field.options?.map((o) => {
            const on = arr.includes(o.value);
            const lbl = L(o.label);
            return (
              <button key={o.value} type="button" onClick={() => toggle(o.value)} className={cn('rounded-full border px-2.5 py-1 text-xs transition-colors', on ? 'border-primary bg-primary text-primary-foreground' : 'border-input bg-background hover:bg-accent')}>
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
      <Input value={value as string} onChange={(e) => onChange(e.target.value)} placeholder={field.placeholder ? L(field.placeholder) : undefined} className={cn('mt-1', invalid && 'border-red-500')} />
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
  const declared = snippet.variables?.map((v) => v.name) ?? detectVariables(snippet.content);
  const finalContent = substitute(snippet.content, values);
  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(finalContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* noop */
    }
  };
  return (
    <Card className="h-full flex flex-col transition-all hover:shadow-md">
      <CardHeader className="pb-2 cursor-pointer" onClick={onToggle}>
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base hover:text-primary transition-colors flex items-center gap-1.5">
            {L(snippet.title)}
            {isCustom && <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[9px] font-bold text-primary">MINE</span>}
          </CardTitle>
          <div className="flex items-center gap-1 shrink-0">
            {isCustom && onEdit && <button onClick={(e) => { e.stopPropagation(); onEdit(); }} aria-label="edit" className="text-muted-foreground hover:text-primary"><Edit3 className="h-4 w-4" /></button>}
            {isCustom && onDelete && <button onClick={(e) => { e.stopPropagation(); onDelete(); }} aria-label="delete" className="text-muted-foreground hover:text-red-500"><Trash2 className="h-4 w-4" /></button>}
            <button onClick={(e) => { e.stopPropagation(); onToggleFav(); }} aria-label="favorite" className="text-muted-foreground hover:text-amber-500"><Star className={cn('h-5 w-5', isFav && 'fill-amber-400 text-amber-400')} /></button>
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
                  <label className="text-xs font-medium">{meta ? L(meta.label) : name}{meta?.default ? '' : ''}</label>
                  <Input value={values[name] ?? meta?.default ?? ''} onChange={(e) => onValueChange(name, e.target.value)} placeholder={name} className="mt-0.5 h-8 text-xs" />
                </div>
              );
            })}
          </div>
        )}
        {expanded && (
          <pre className="whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-3 text-xs font-mono leading-relaxed max-h-[40vh] overflow-y-auto mb-3">{finalContent}</pre>
        )}
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={onClose}>
      <div className="w-full max-w-lg rounded-lg border bg-background p-5 max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-bold">{lang === 'ko' ? (initial ? '내 프롬프트 수정' : '내 프롬프트 추가') : (initial ? 'Edit my prompt' : 'Add my prompt')}</h3>
          <button onClick={onClose} aria-label="close"><X className="h-5 w-5 text-muted-foreground" /></button>
        </div>
        <div className="space-y-3">
          <div>
            <label className="text-sm font-medium">{lang === 'ko' ? '제목' : 'Title'}<span className="text-red-500">*</span></label>
            <Input value={title} onChange={(e) => setTitle(e.target.value)} className="mt-1" />
          </div>
          <div>
            <label className="text-sm font-medium">{lang === 'ko' ? '본문' : 'Content'}<span className="text-red-500">*</span></label>
            <textarea value={content} onChange={(e) => setContent(e.target.value)} className="mt-1 flex min-h-[160px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" />
            <p className="text-xs text-muted-foreground mt-1">{lang === 'ko' ? '{{변수}} 문법으로 변수를 넣으면 확장 시 채울 수 있습니다.' : 'Use {{variable}} placeholders to fill in later.'}</p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-sm font-medium">{lang === 'ko' ? '분야' : 'Category'}</label>
              <select value={category} onChange={(e) => setCategory(e.target.value as PromptCategory)} className="mt-1 flex h-10 w-full rounded-md border border-input bg-background px-3 text-sm">
                {categories.map((c) => <option key={c.id} value={c.id}>{L(c.label)}</option>)}
              </select>
            </div>
            <div>
              <label className="text-sm font-medium">{lang === 'ko' ? '태그 (쉼표)' : 'Tags (comma)'}</label>
              <Input value={tags} onChange={(e) => setTags(e.target.value)} className="mt-1" />
            </div>
          </div>
        </div>
        <div className="flex justify-end gap-2 mt-5">
          <Button variant="outline" size="sm" onClick={onClose}>{lang === 'ko' ? '취소' : 'Cancel'}</Button>
          <Button size="sm" onClick={submit} disabled={!title.trim() || !content.trim()}>{lang === 'ko' ? '저장' : 'Save'}</Button>
        </div>
      </div>
    </div>
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
