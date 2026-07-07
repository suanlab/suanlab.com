'use client';

import { useState, useMemo, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  Search, Copy, Check, Download, Star, ArrowLeft, Wand2, Lightbulb,
  ShieldCheck, FileSearch, Compass, FlaskConical, PenLine, Target,
  MessageSquareReply, Library, FileText, Database, Presentation,
  GraduationCap, RotateCcw, ExternalLink,
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
} from '@/data/prompts';

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

export default function PromptsClient() {
  const builders = promptBuilders;
  const snippets = promptSnippets;
  const categories = promptCategories;
  const { language } = useLanguage();
  const L = (t: LocalizedText) => t[language];

  const [tab, setTab] = useState<'builders' | 'library' | 'favorites'>('builders');
  const [query, setQuery] = useState('');
  const [selectedCats, setSelectedCats] = useState<Set<PromptCategory>>(new Set());
  const [activeBuilder, setActiveBuilder] = useState<PromptBuilder | null>(null);
  const [favorites, setFavorites] = useState<string[]>([]);
  const [expandedSnip, setExpandedSnip] = useState<string | null>(null);

  // favorites from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem(FAV_KEY);
      if (raw) setFavorites(JSON.parse(raw));
    } catch {
      /* noop */
    }
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

  const toggleCat = (c: PromptCategory) => {
    setSelectedCats((prev) => {
      const next = new Set(prev);
      if (next.has(c)) next.delete(c);
      else next.add(c);
      return next;
    });
  };

  const q = query.trim().toLowerCase();
  const matches = (hay: string) => !q || hay.toLowerCase().includes(q);
  const catOk = (c: PromptCategory) => selectedCats.size === 0 || selectedCats.has(c);

  const filteredBuilders = useMemo(
    () =>
      builders.filter(
        (b) =>
          catOk(b.category) &&
          matches(`${L(b.title)} ${L(b.description)} ${b.tags.join(' ')}`),
      ),
    [builders, q, selectedCats, language],
  );

  const filteredSnippets = useMemo(
    () =>
      snippets.filter(
        (s) =>
          catOk(s.category) &&
          matches(`${L(s.title)} ${L(s.description)} ${s.tags.join(' ')} ${s.content}`),
      ),
    [snippets, q, selectedCats, language],
  );

  const favBuilders = filteredBuilders.filter((b) => favorites.includes(b.id));
  const favSnippets = filteredSnippets.filter((s) => favorites.includes(s.id));

  const catLabel = (c: PromptCategory) => L(categories.find((x) => x.id === c)!.label);
  const catColor = (c: PromptCategory) => categories.find((x) => x.id === c)!.color;

  const renderCategoryBar = () => (
    <div className="flex flex-wrap gap-2 mb-4">
      {categories.map((c) => {
        const active = selectedCats.has(c.id);
        return (
          <Button
            key={c.id}
            variant={active ? 'default' : 'outline'}
            size="sm"
            onClick={() => toggleCat(c.id)}
            className={cn(!active && 'hover:bg-accent')}
          >
            {L(c.label)}
          </Button>
        );
      })}
      {selectedCats.size > 0 && (
        <Button variant="ghost" size="sm" onClick={() => setSelectedCats(new Set())}>
          <RotateCcw className="mr-1 h-3 w-3" />
          {language === 'ko' ? '초기화' : 'Reset'}
        </Button>
      )}
    </div>
  );

  return (
    <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)} className="w-full">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
        <TabsList>
          <TabsTrigger value="builders">
            <Wand2 className="mr-1.5 h-4 w-4" />
            {language === 'ko' ? '빌더' : 'Builders'} ({builders.length})
          </TabsTrigger>
          <TabsTrigger value="library">
            <Library className="mr-1.5 h-4 w-4" />
            {language === 'ko' ? '라이브러리' : 'Library'} ({snippets.length})
          </TabsTrigger>
          <TabsTrigger value="favorites">
            <Star className="mr-1.5 h-4 w-4" />
            {language === 'ko' ? '즐겨찾기' : 'Favorites'} ({favorites.length})
          </TabsTrigger>
        </TabsList>
      </div>

      {/* Search + categories (hidden when a builder detail is open) */}
      {!activeBuilder && (
        <>
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder={language === 'ko' ? '프롬프트 검색...' : 'Search prompts...'}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="pl-9"
            />
          </div>
          {renderCategoryBar()}
        </>
      )}

      <TabsContent value="builders" className="mt-0">
        {activeBuilder ? (
          <BuilderDetail
            builder={activeBuilder}
            onBack={() => setActiveBuilder(null)}
            isFav={favorites.includes(activeBuilder.id)}
            onToggleFav={() => toggleFav(activeBuilder.id)}
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
                  onOpen={() => {
                    setActiveBuilder(b);
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                  }}
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
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredSnippets.map((s) => (
            <SnippetCard
              key={s.id}
              snippet={s}
              expanded={expandedSnip === s.id}
              onToggle={() => setExpandedSnip(expandedSnip === s.id ? null : s.id)}
              isFav={favorites.includes(s.id)}
              onToggleFav={() => toggleFav(s.id)}
              catLabel={catLabel(s.category)}
              catColor={catColor(s.category)}
              L={L}
              lang={language}
            />
          ))}
          {filteredSnippets.length === 0 && <EmptyState lang={language} />}
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
                <p className="text-sm font-medium text-muted-foreground mb-3 mt-2">
                  {language === 'ko' ? '빌더' : 'Builders'}
                </p>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 mb-8">
                  {favBuilders.map((b) => {
                    const Icon = ICONS[b.icon] ?? Wand2;
                    return (
                      <BuilderCard
                        key={b.id}
                        builder={b}
                        Icon={Icon}
                        isFav
                        onToggleFav={() => toggleFav(b.id)}
                        onOpen={() => {
                          setTab('builders');
                          setActiveBuilder(b);
                          window.scrollTo({ top: 0, behavior: 'smooth' });
                        }}
                        catLabel={catLabel(b.category)}
                        catColor={catColor(b.category)}
                        L={L}
                        lang={language}
                      />
                    );
                  })}
                </div>
              </>
            )}
            {favSnippets.length > 0 && (
              <>
                <p className="text-sm font-medium text-muted-foreground mb-3">
                  {language === 'ko' ? '라이브러리' : 'Library'}
                </p>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {favSnippets.map((s) => (
                    <SnippetCard
                      key={s.id}
                      snippet={s}
                      expanded={expandedSnip === s.id}
                      onToggle={() => setExpandedSnip(expandedSnip === s.id ? null : s.id)}
                      isFav
                      onToggleFav={() => toggleFav(s.id)}
                      catLabel={catLabel(s.category)}
                      catColor={catColor(s.category)}
                      L={L}
                      lang={language}
                    />
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </TabsContent>
    </Tabs>
  );
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
          <div className={cn('inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary shrink-0')}>
            <Icon className="h-5 w-5" />
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); onToggleFav(); }}
            aria-label="favorite"
            className="text-muted-foreground hover:text-amber-500 transition-colors"
          >
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

// ─── Builder detail (form + live output) ───
function BuilderDetail({
  builder, onBack, isFav, onToggleFav, language, L,
}: {
  builder: PromptBuilder;
  onBack: () => void;
  isFav: boolean;
  onToggleFav: () => void;
  language: 'ko' | 'en';
  L: (t: LocalizedText) => string;
}) {
  const [values, setValues] = useState<Record<string, string | string[]>>(() => {
    const v: Record<string, string | string[]> = {};
    for (const f of builder.fields) {
      if (f.type === 'multiselect') v[f.id] = [];
      else if (f.default) v[f.id] = f.default as string;
      else v[f.id] = '';
    }
    return v;
  });
  const [copied, setCopied] = useState(false);
  const [missing, setMissing] = useState<Set<string>>(new Set());

  // reset when builder changes
  useEffect(() => {
    const v: Record<string, string | string[]> = {};
    for (const f of builder.fields) {
      if (f.type === 'multiselect') v[f.id] = [];
      else if (f.default) v[f.id] = f.default as string;
      else v[f.id] = '';
    }
    setValues(v);
    setMissing(new Set());
  }, [builder.id]);

  const prompt = useMemo(() => {
    try {
      return builder.generate(values);
    } catch {
      return '';
    }
  }, [builder, values]);

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
    const requiredMissing = builder.fields.filter((f) => f.required && f.type !== 'multiselect' && !String(values[f.id] ?? '').trim());
    const multiMissing = builder.fields.filter((f) => f.required && f.type === 'multiselect' && (values[f.id] as string[]).length === 0);
    const allMissing = [...requiredMissing, ...multiMissing];
    if (allMissing.length > 0) {
      setMissing(new Set(allMissing.map((f) => f.id)));
      return;
    }
    try {
      await navigator.clipboard.writeText(prompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* noop */
    }
  };

  const handleDownload = () => {
    const blob = new Blob([`# ${L(builder.title)}\n\n${prompt}\n`], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${builder.id}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="flex items-center justify-between gap-2 mb-4">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="mr-1.5 h-4 w-4" />
          {language === 'ko' ? '목록으로' : 'Back to list'}
        </Button>
        <Button variant="ghost" size="sm" onClick={onToggleFav}>
          <Star className={cn('mr-1.5 h-4 w-4', isFav && 'fill-amber-400 text-amber-400')} />
          {isFav ? (language === 'ko' ? '즐겨찾기됨' : 'Favorited') : (language === 'ko' ? '즐겨찾기' : 'Favorite')}
        </Button>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Form */}
        <div>
          <h2 className="text-xl font-bold mb-1">{L(builder.title)}</h2>
          <p className="text-sm text-muted-foreground mb-5">{L(builder.description)}</p>
          <div className="space-y-4">
            {builder.fields.map((f) => (
              <FieldInput
                key={f.id}
                field={f}
                value={values[f.id]}
                onChange={(v) => setValue(f.id, v)}
                invalid={missing.has(f.id)}
                L={L}
              />
            ))}
          </div>
        </div>

        {/* Output */}
        <div className="lg:sticky lg:top-24 self-start">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-muted-foreground">
              {language === 'ko' ? '생성된 프롬프트' : 'Generated prompt'}
            </span>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={handleDownload} disabled={!prompt}>
                <Download className="mr-1 h-3.5 w-3.5" />
                .md
              </Button>
              <Button size="sm" onClick={handleCopy}>
                {copied ? <Check className="mr-1 h-3.5 w-3.5 text-green-500" /> : <Copy className="mr-1 h-3.5 w-3.5" />}
                {copied ? (language === 'ko' ? '복사됨!' : 'Copied!') : (language === 'ko' ? '복사' : 'Copy')}
              </Button>
            </div>
          </div>
          {missing.size > 0 && (
            <p className="text-xs text-red-500 mb-2">
              {language === 'ko' ? '필수 항목을 입력해 주세요.' : 'Please fill required fields.'}
            </p>
          )}
          <pre className="whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-4 text-xs font-mono leading-relaxed max-h-[60vh] overflow-y-auto">
            {prompt || (language === 'ko' ? '왼쪽 폼을 채우면 프롬프트가 생성됩니다.' : 'Fill the form to generate the prompt.')}
          </pre>
          <p className="mt-2 text-[11px] text-muted-foreground flex items-center gap-1">
            <ExternalLink className="h-3 w-3" />
            {language === 'ko'
              ? '복사한 프롬프트를 Claude CLI 등에 붙여넣어 사용하세요.'
              : 'Paste the copied prompt into Claude CLI or your AI assistant.'}
          </p>
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
          className={cn(
            'mt-1 flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
            invalid && 'border-red-500',
          )}
        />
        {help}
      </div>
    );
  }

  if (field.type === 'select') {
    return (
      <div>
        {label}
        <select
          value={value as string}
          onChange={(e) => onChange(e.target.value)}
          className={cn(
            'mt-1 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
            invalid && 'border-red-500',
          )}
        >
          {field.options?.map((o) => (
            <option key={o.value} value={o.value}>{L(o.label)}</option>
          ))}
        </select>
        {help}
      </div>
    );
  }

  if (field.type === 'multiselect') {
    const arr = value as string[];
    const toggle = (val: string) =>
      onChange(arr.includes(val) ? arr.filter((x) => x !== val) : [...arr, val]);
    return (
      <div>
        {label}
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {field.options?.map((o) => {
            const on = arr.includes(o.value);
            return (
              <button
                key={o.value}
                type="button"
                onClick={() => toggle(o.value)}
                className={cn(
                  'rounded-full border px-2.5 py-1 text-xs transition-colors',
                  on
                    ? 'border-primary bg-primary text-primary-foreground'
                    : 'border-input bg-background hover:bg-accent',
                )}
              >
                {o.value === o.value && L(o.label).length > 42 ? L(o.label).slice(0, 40) + '…' : L(o.label)}
              </button>
            );
          })}
        </div>
        {help}
      </div>
    );
  }

  // text
  return (
    <div>
      {label}
      <Input
        value={value as string}
        onChange={(e) => onChange(e.target.value)}
        placeholder={field.placeholder ? L(field.placeholder) : undefined}
        className={cn('mt-1', invalid && 'border-red-500')}
      />
      {help}
    </div>
  );
}

// ─── Snippet card ───
function SnippetCard({
  snippet, expanded, onToggle, isFav, onToggleFav, catLabel, catColor, L, lang,
}: {
  snippet: PromptSnippet;
  expanded: boolean;
  onToggle: () => void;
  isFav: boolean;
  onToggleFav: () => void;
  catLabel: string;
  catColor: string;
  L: (t: LocalizedText) => string;
  lang: 'ko' | 'en';
}) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(snippet.content);
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
          <CardTitle className="text-base hover:text-primary transition-colors">{L(snippet.title)}</CardTitle>
          <button
            onClick={(e) => { e.stopPropagation(); onToggleFav(); }}
            aria-label="favorite"
            className="text-muted-foreground hover:text-amber-500 transition-colors shrink-0"
          >
            <Star className={cn('h-5 w-5', isFav && 'fill-amber-400 text-amber-400')} />
          </button>
        </div>
        <p className="text-sm text-muted-foreground line-clamp-2">{L(snippet.description)}</p>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col">
        <div className="flex flex-wrap gap-1.5 mb-3">
          <span className={cn('inline-flex items-center rounded px-2 py-0.5 text-[10px] font-medium', catColor)}>{catLabel}</span>
          {snippet.tags.slice(0, 2).map((t) => (
            <span key={t} className="inline-flex items-center rounded bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">#{t}</span>
          ))}
        </div>
        {expanded && (
          <pre className="whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-3 text-xs font-mono leading-relaxed max-h-[40vh] overflow-y-auto mb-3">
            {snippet.content}
          </pre>
        )}
        <div className="mt-auto flex gap-2">
          <Button className="flex-1" size="sm" variant={expanded ? 'outline' : 'default'} onClick={onToggle}>
            {expanded ? (lang === 'ko' ? '접기' : 'Collapse') : (lang === 'ko' ? '보기' : 'View')}
          </Button>
          <Button size="sm" variant="outline" onClick={handleCopy}>
            {copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}
            <span className="ml-1">{copied ? (lang === 'ko' ? '복사됨' : 'Copied') : (lang === 'ko' ? '복사' : 'Copy')}</span>
          </Button>
        </div>
      </CardContent>
    </Card>
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
