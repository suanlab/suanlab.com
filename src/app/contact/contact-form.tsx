'use client';

import { site } from '@/config/site';
import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react';
import { Copy, ExternalLink, Mail } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';

const inputClass = 'w-full px-4 py-2 rounded-lg border border-input bg-background text-foreground placeholder:text-muted-foreground focus:ring-2 focus:ring-primary';

export default function ContactForm() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const recipient = site.contact.emails[0];
  const formRef = useRef<HTMLFormElement>(null);
  const copyRef = useRef<HTMLTextAreaElement>(null);
  const [formData, setFormData] = useState({ name: '', email: '', subject: '', message: '' });
  const [notice, setNotice] = useState('');
  const [composeRequested, setComposeRequested] = useState(false);
  const [manualCopy, setManualCopy] = useState(false);
  const subject = formData.subject.trim() || 'Contact from SuanLab';
  const body = `Name: ${formData.name.trim()}\nEmail: ${formData.email.trim()}\n\n${formData.message.trim()}`;
  const draft = `To: ${recipient}\nSubject: ${subject}\n\n${body}`;
  const gmail = new URL('https://mail.google.com/mail/');
  gmail.search = new URLSearchParams({ view: 'cm', fs: '1', to: recipient, su: subject, body }).toString();
  const mailto = `mailto:${recipient}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

  useEffect(() => {
    if (manualCopy) { copyRef.current?.focus(); copyRef.current?.select(); }
  }, [manualCopy]);

  function handleChange(event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
    setFormData(current => ({ ...current, [event.target.name]: event.target.value }));
    setNotice('');
  }

  function validate() {
    if (!formRef.current?.reportValidity()) return false;
    if (Object.values(formData).some(value => !value.trim())) {
      setNotice(ko ? '각 항목에 공백이 아닌 내용을 입력해 주세요.' : 'Please enter text in every field.');
      return false;
    }
    return true;
  }

  function openGmail(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!validate()) return;
    window.open(gmail.toString(), '_blank', 'noopener,noreferrer');
    setComposeRequested(true);
    setNotice(ko ? 'Gmail 작성 창에서 내용을 확인한 뒤 전송해 주세요. 창이 열리지 않으면 아래 링크를 이용하세요.' : 'Review and send your message in Gmail. If the compose window does not open, use the link below.');
  }

  function openEmailApp() {
    if (!validate()) return;
    window.location.href = mailto;
    setNotice(ko ? '이메일 앱에서 내용을 확인한 뒤 전송해 주세요. 앱이 열리지 않으면 Gmail 작성 버튼을 이용하세요.' : 'Review and send your message in your email app. If it does not open, use Gmail instead.');
  }

  async function copyDraft() {
    if (!validate()) return;
    try {
      await navigator.clipboard.writeText(draft);
      setNotice(ko ? '문의 내용을 복사했습니다. 사용하는 메일 서비스에 붙여넣고 전송해 주세요.' : 'Draft copied. Paste it into your email service and send it.');
    } catch {
      setManualCopy(true);
      setNotice(ko ? '아래 문의 내용을 선택해 복사한 뒤 메일에 붙여넣어 주세요.' : 'Select and copy the draft below, then paste it into your email.');
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{ko ? '문의 메일 작성' : 'Write a contact email'}</CardTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">{ko ? 'Gmail 웹 또는 이메일 앱에서 작성한 내용을 확인하고 전송할 수 있습니다. Gmail을 사용하지 않으면 문의 내용을 복사해 다른 메일 서비스에 붙여넣으세요.' : 'Review and send your message using Gmail on the web or your email app. You can also copy the draft into another email service.'}</p>
        <p className="text-sm">{ko ? '받는 사람: ' : 'To: '}<a href={`mailto:${recipient}`} className="text-primary underline">{recipient}</a></p>
      </CardHeader>
      <CardContent>
        <form ref={formRef} onSubmit={openGmail} className="space-y-6">
          <div>
            <label htmlFor="name" className="mb-2 block text-sm font-medium">{ko ? '이름 *' : 'Name *'}</label>
            <input type="text" id="name" name="name" autoComplete="name" value={formData.name} onChange={handleChange} required maxLength={120} className={inputClass} placeholder={ko ? '이름을 입력해 주세요' : 'Your name'} />
          </div>
          <div>
            <label htmlFor="email" className="mb-2 block text-sm font-medium">{ko ? '이메일 *' : 'Email *'}</label>
            <input type="email" id="email" name="email" autoComplete="email" value={formData.email} onChange={handleChange} required maxLength={254} className={inputClass} placeholder="your.email@example.com" />
          </div>
          <div>
            <label htmlFor="subject" className="mb-2 block text-sm font-medium">{ko ? '제목 *' : 'Subject *'}</label>
            <input type="text" id="subject" name="subject" value={formData.subject} onChange={handleChange} required maxLength={200} className={inputClass} placeholder={ko ? '문의 제목을 입력해 주세요' : 'Message subject'} />
          </div>
          <div>
            <label htmlFor="message" className="mb-2 block text-sm font-medium">{ko ? '메시지 *' : 'Message *'}</label>
            <textarea id="message" name="message" value={formData.message} onChange={handleChange} required maxLength={4000} rows={6} className={`${inputClass} resize-y`} placeholder={ko ? '문의 내용을 입력해 주세요' : 'Your message…'} />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <Button type="submit"><ExternalLink aria-hidden="true" className="mr-2 h-4 w-4" />{ko ? 'Gmail에서 메일 작성' : 'Compose in Gmail'}</Button>
            <Button type="button" variant="outline" onClick={openEmailApp}><Mail aria-hidden="true" className="mr-2 h-4 w-4" />{ko ? '이메일 앱에서 작성' : 'Compose in email app'}</Button>
          </div>
          <Button type="button" variant="outline" className="w-full" onClick={copyDraft}><Copy aria-hidden="true" className="mr-2 h-4 w-4" />{ko ? '문의 내용 복사' : 'Copy email draft'}</Button>
          <p className="text-xs leading-relaxed text-muted-foreground">{ko ? '* 필수 항목입니다. 메일 작성 창에서 전송 버튼을 눌러야 문의가 전달됩니다.' : '* Required fields. Your message is sent when you press Send in the email compose window.'}</p>
          {notice && <p role="status" className="rounded-lg border bg-muted/30 p-4 text-sm leading-relaxed">{notice}</p>}
          {composeRequested && <a href={gmail.toString()} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 text-sm text-primary underline">{ko ? 'Gmail 작성 창 직접 열기' : 'Open Gmail compose window'}<ExternalLink aria-hidden="true" className="h-4 w-4" /></a>}
          {manualCopy && <div><label htmlFor="email-draft" className="mb-2 block text-sm font-medium">{ko ? '복사할 문의 내용' : 'Email draft to copy'}</label><textarea ref={copyRef} id="email-draft" readOnly value={draft} rows={8} className={`${inputClass} resize-y`} onFocus={event => event.target.select()} /></div>}
        </form>
      </CardContent>
    </Card>
  );
}
