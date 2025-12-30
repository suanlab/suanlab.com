'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Save, FileText, Code, Loader2 } from 'lucide-react';

interface PreviewPaneProps {
  content: string | null;
  onSave: () => void;
  isGenerating: boolean;
}

export default function PreviewPane({
  content,
  onSave,
  isGenerating,
}: PreviewPaneProps) {
  const [editedContent, setEditedContent] = useState(content || '');
  const [isSaving, setIsSaving] = useState(false);

  // Update edited content when new content is generated
  if (content && content !== editedContent && !isGenerating) {
    setEditedContent(content);
  }

  const handleSave = async () => {
    setIsSaving(true);
    await onSave();
    setIsSaving(false);
  };

  if (isGenerating) {
    return (
      <Card className="h-full min-h-[500px] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-lg font-medium">Generating content...</p>
          <p className="text-sm text-muted-foreground">
            This may take a minute
          </p>
        </div>
      </Card>
    );
  }

  if (!content) {
    return (
      <Card className="h-full min-h-[500px] flex items-center justify-center">
        <div className="text-center text-muted-foreground">
          <FileText className="h-12 w-12 mx-auto mb-4" />
          <p>Generated content will appear here</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Preview</CardTitle>
        <Button onClick={handleSave} disabled={isSaving}>
          {isSaving ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Saving...
            </>
          ) : (
            <>
              <Save className="mr-2 h-4 w-4" />
              Save Post
            </>
          )}
        </Button>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="preview">
          <TabsList className="mb-4">
            <TabsTrigger value="preview">
              <FileText className="h-4 w-4 mr-1.5" />
              Preview
            </TabsTrigger>
            <TabsTrigger value="source">
              <Code className="h-4 w-4 mr-1.5" />
              Source
            </TabsTrigger>
          </TabsList>

          <TabsContent value="preview" className="min-h-[400px]">
            <div className="prose dark:prose-invert max-w-none overflow-auto max-h-[600px] p-4 border rounded-lg bg-background">
              <div
                dangerouslySetInnerHTML={{
                  __html: simpleMarkdownToHtml(editedContent),
                }}
              />
            </div>
          </TabsContent>

          <TabsContent value="source" className="min-h-[400px]">
            <textarea
              value={editedContent}
              onChange={(e) => setEditedContent(e.target.value)}
              className="w-full h-[600px] p-4 font-mono text-sm border rounded-lg bg-background resize-none"
              placeholder="Markdown content..."
            />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

/**
 * Simple markdown to HTML conversion for preview
 * (Full rendering happens server-side with remark/rehype)
 */
function simpleMarkdownToHtml(markdown: string): string {
  return (
    markdown
      // Headers
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^# (.+)$/gm, '<h1>$1</h1>')
      // Bold
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // Italic
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      // Code blocks
      .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
      // Inline code
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      // Links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
      // Lists
      .replace(/^- (.+)$/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
      // Paragraphs
      .replace(/\n\n/g, '</p><p>')
      .replace(/^(.+)$/gm, (match) => {
        if (
          match.startsWith('<') ||
          match.startsWith('#') ||
          match.startsWith('-')
        ) {
          return match;
        }
        return `<p>${match}</p>`;
      })
  );
}
