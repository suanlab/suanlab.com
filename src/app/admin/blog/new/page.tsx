'use client';

import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import TopicForm from '@/components/admin/TopicForm';
import PaperUpload from '@/components/admin/PaperUpload';
import PreviewPane from '@/components/admin/PreviewPane';

export default function NewBlogPost() {
  const [generatedContent, setGeneratedContent] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = (content: string) => {
    setGeneratedContent(content);
    setError(null);
  };

  const handleError = (err: string) => {
    setError(err);
    setGeneratedContent(null);
  };

  const handleSave = async () => {
    if (!generatedContent) return;

    try {
      const response = await fetch('/api/admin/blog/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: generatedContent }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to save');
      }

      alert(`Saved successfully! Path: ${data.path}`);
      setGeneratedContent(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed');
    }
  };

  return (
    <div className="container py-8">
      <h1 className="text-3xl font-bold mb-8">Create New Blog Post</h1>

      <div className="grid gap-8 lg:grid-cols-2">
        {/* Input Forms */}
        <div>
          <Tabs defaultValue="topic">
            <TabsList className="mb-4">
              <TabsTrigger value="topic">Topic-Based</TabsTrigger>
              <TabsTrigger value="paper">Paper Summary</TabsTrigger>
            </TabsList>

            <TabsContent value="topic">
              <TopicForm
                onGenerate={handleGenerate}
                onError={handleError}
                isGenerating={isGenerating}
                setIsGenerating={setIsGenerating}
              />
            </TabsContent>

            <TabsContent value="paper">
              <PaperUpload
                onGenerate={handleGenerate}
                onError={handleError}
                isGenerating={isGenerating}
                setIsGenerating={setIsGenerating}
              />
            </TabsContent>
          </Tabs>

          {error && (
            <div className="mt-4 p-4 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
              {error}
            </div>
          )}
        </div>

        {/* Preview */}
        <div>
          <PreviewPane
            content={generatedContent}
            onSave={handleSave}
            isGenerating={isGenerating}
          />
        </div>
      </div>
    </div>
  );
}
