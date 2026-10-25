import { parseFragment, serialize, type DefaultTreeAdapterMap } from 'parse5';

/** Localize human-readable markup without changing code, formulas, IDs or URLs. */
export function localizedMarkup(html: string, translate: (text: string) => string): string {
  const fragment = parseFragment(html);
  const visit = (node: DefaultTreeAdapterMap['node'], source = false) => {
    const tag = 'tagName' in node ? node.tagName : '';
    const original = 'attrs' in node && node.attrs.some(attribute => (attribute.name === 'lang' && attribute.value === 'ko') || attribute.name === 'data-original-language');
    const preserve = source || original || ['script', 'style', 'pre', 'code', 'math', 'svg'].includes(tag);
    if (!preserve && 'value' in node) node.value = translate(node.value);
    if (!preserve && 'attrs' in node) for (const attribute of node.attrs) {
      if (['alt', 'title', 'aria-label'].includes(attribute.name)) attribute.value = translate(attribute.value);
    }
    if ('childNodes' in node) node.childNodes.forEach(child => visit(child, preserve));
  };
  visit(fragment);
  return serialize(fragment);
}
