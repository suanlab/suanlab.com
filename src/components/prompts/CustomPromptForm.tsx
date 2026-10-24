'use client';

import { useState, useId } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { X } from 'lucide-react';
import { type PromptCategory, type LocalizedText } from '@/data/prompts';
import type { CustomSnippet } from './types';

export default function CustomPromptForm({
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
