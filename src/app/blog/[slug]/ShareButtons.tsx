'use client';
import { site } from '@/config/site';

import { useLanguage } from '@/components/language-provider';
import { useState } from 'react';
import { Share2, Facebook, Linkedin, LinkIcon, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';

// X (Twitter) icon component
const XIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
  </svg>
);

interface ShareButtonsProps {
  title: string;
  slug: string;
}

const BASE_URL = site.url;

export default function ShareButtons({ title, slug }: ShareButtonsProps) {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [copied, setCopied] = useState(false);

  const url = `${BASE_URL}/blog/${slug}/`;

  const shareLinks = {
    twitter: `https://twitter.com/intent/tweet?url=${encodeURIComponent(url)}&text=${encodeURIComponent(title)}`,
    facebook: `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`,
    linkedin: `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}`,
  };

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy link:', err);
    }
  };

  return (
    <div className="flex items-center gap-3">
      <span className="flex items-center gap-1 text-sm text-muted-foreground">
        <Share2 className="h-4 w-4" />
        {ko ? '공유하기' : 'Share'}
      </span>
      <div className="flex items-center gap-2">
        <a
          href={shareLinks.twitter}
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-full hover:bg-muted transition-colors"
          aria-label={ko ? 'X (Twitter)에 공유' : 'Share on X (Twitter)'}
        >
          <XIcon className="h-4 w-4" />
        </a>
        <a
          href={shareLinks.facebook}
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-full hover:bg-muted transition-colors"
          aria-label={ko ? 'Facebook에 공유' : 'Share on Facebook'}
        >
          <Facebook className="h-4 w-4" />
        </a>
        <a
          href={shareLinks.linkedin}
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-full hover:bg-muted transition-colors"
          aria-label={ko ? 'LinkedIn에 공유' : 'Share on LinkedIn'}
        >
          <Linkedin className="h-4 w-4" />
        </a>
        <Button
          variant="ghost"
          size="icon"
          onClick={handleCopyLink}
          className="h-8 w-8"
          aria-label={ko ? '링크 복사' : 'Copy link'}
        >
          {copied ? (
            <Check className="h-4 w-4 text-green-500" />
          ) : (
            <LinkIcon className="h-4 w-4" />
          )}
        </Button>
      </div>
    </div>
  );
}
