'use client';

import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Loader2, Upload } from 'lucide-react';

interface PaperUploadProps {
  onGenerate: (content: string) => void;
  onError: (error: string) => void;
  isGenerating: boolean;
  setIsGenerating: (v: boolean) => void;
}

export default function PaperUpload({
  onGenerate,
  onError,
  isGenerating,
  setIsGenerating,
}: PaperUploadProps) {
  const [arxivId, setArxivId] = useState('');
  const [pdfUrl, setPdfUrl] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleArxivSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);

    try {
      const response = await fetch('/api/admin/blog/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'paper', arxivId }),
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

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);

    try {
      const response = await fetch('/api/admin/blog/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'paper', pdfUrl }),
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

  const handleFileSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setIsGenerating(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', 'paper');

    try {
      const response = await fetch('/api/admin/blog/generate', {
        method: 'POST',
        body: formData,
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
        <CardTitle>Paper-Based Content Generation</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="arxiv">
          <TabsList className="mb-4">
            <TabsTrigger value="arxiv">arXiv</TabsTrigger>
            <TabsTrigger value="url">PDF URL</TabsTrigger>
            <TabsTrigger value="file">File Upload</TabsTrigger>
          </TabsList>

          <TabsContent value="arxiv">
            <form onSubmit={handleArxivSubmit} className="space-y-4">
              <div>
                <label
                  htmlFor="arxiv"
                  className="block text-sm font-medium mb-1.5"
                >
                  arXiv ID or URL
                </label>
                <input
                  id="arxiv"
                  type="text"
                  value={arxivId}
                  onChange={(e) => setArxivId(e.target.value)}
                  placeholder="e.g., 2401.12345 or https://arxiv.org/abs/2401.12345"
                  className="w-full p-2.5 border rounded-md bg-background"
                />
              </div>
              <Button type="submit" disabled={isGenerating || !arxivId}>
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Generate Summary'
                )}
              </Button>
            </form>
          </TabsContent>

          <TabsContent value="url">
            <form onSubmit={handleUrlSubmit} className="space-y-4">
              <div>
                <label
                  htmlFor="pdfUrl"
                  className="block text-sm font-medium mb-1.5"
                >
                  PDF URL
                </label>
                <input
                  id="pdfUrl"
                  type="url"
                  value={pdfUrl}
                  onChange={(e) => setPdfUrl(e.target.value)}
                  placeholder="https://example.com/paper.pdf"
                  className="w-full p-2.5 border rounded-md bg-background"
                />
              </div>
              <Button type="submit" disabled={isGenerating || !pdfUrl}>
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Generate Summary'
                )}
              </Button>
            </form>
          </TabsContent>

          <TabsContent value="file">
            <form onSubmit={handleFileSubmit} className="space-y-4">
              <div>
                <label
                  htmlFor="file"
                  className="block text-sm font-medium mb-1.5"
                >
                  PDF File
                </label>
                <div
                  className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:border-primary transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    id="file"
                    type="file"
                    accept=".pdf"
                    ref={fileInputRef}
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                    className="hidden"
                  />
                  <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                  {file ? (
                    <p className="text-sm">{file.name}</p>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      Click to upload PDF
                    </p>
                  )}
                </div>
              </div>
              <Button type="submit" disabled={isGenerating || !file}>
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Generate Summary'
                )}
              </Button>
            </form>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
