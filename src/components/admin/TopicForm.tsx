'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2 } from 'lucide-react';

interface TopicFormProps {
  onGenerate: (content: string) => void;
  onError: (error: string) => void;
  isGenerating: boolean;
  setIsGenerating: (v: boolean) => void;
}

const CATEGORIES = [
  'Deep Learning',
  'Machine Learning',
  'NLP',
  'Computer Vision',
  'Data Science',
  'General',
];

export default function TopicForm({
  onGenerate,
  onError,
  isGenerating,
  setIsGenerating,
}: TopicFormProps) {
  const [topic, setTopic] = useState('');
  const [category, setCategory] = useState('General');
  const [tags, setTags] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);

    try {
      const response = await fetch('/api/admin/blog/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'topic',
          topic,
          category,
          tags: tags
            .split(',')
            .map((t) => t.trim())
            .filter(Boolean),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Generation failed');
      }

      onGenerate(data.content);
    } catch (err) {
      onError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Topic-Based Content Generation</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="topic"
              className="block text-sm font-medium mb-1.5"
            >
              Topic / Keywords
            </label>
            <input
              id="topic"
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g., Transformer Self-Attention Mechanism"
              className="w-full p-2.5 border rounded-md bg-background"
              required
            />
          </div>

          <div>
            <label
              htmlFor="category"
              className="block text-sm font-medium mb-1.5"
            >
              Category
            </label>
            <select
              id="category"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full p-2.5 border rounded-md bg-background"
            >
              {CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="tags" className="block text-sm font-medium mb-1.5">
              Tags (comma separated)
            </label>
            <input
              id="tags"
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="e.g., Transformer, Attention, NLP"
              className="w-full p-2.5 border rounded-md bg-background"
            />
          </div>

          <Button type="submit" disabled={isGenerating || !topic}>
            {isGenerating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              'Generate Content'
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
