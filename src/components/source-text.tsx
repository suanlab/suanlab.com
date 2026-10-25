export default function SourceText({ text }: { text: string }) {
  const korean = /[가-힣]/.test(text);
  return <span lang={korean ? 'ko' : undefined} data-original-language={korean ? 'ko' : undefined}>{text}</span>;
}
