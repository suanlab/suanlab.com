import type { PromptWorkflow } from '@/data/prompts';
const WF_KEY = 'suanlab-prompt-workflow-state';

export function loadWorkflowState(wfId: string): { outputs: string[] } {
  try {
    const raw = localStorage.getItem(WF_KEY);
    const all = raw ? JSON.parse(raw) : {};
    const outputs = all?.[wfId]?.outputs;
    return { outputs: Array.isArray(outputs) && outputs.every(value => typeof value === 'string') ? outputs : [] };
  } catch {
    return { outputs: [] };
  }
}
export function saveWorkflowOutput(ctx: { wf: PromptWorkflow; step: number }, output: string) {
  try {
    const raw = localStorage.getItem(WF_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    const all = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
    const prev = loadWorkflowState(ctx.wf.id).outputs;
    const outputs = prev.slice(0, ctx.step);
    outputs[ctx.step] = output;
    all[ctx.wf.id] = { outputs };
    localStorage.setItem(WF_KEY, JSON.stringify(all));
    return true;
  } catch {
    return false;
  }
}
export function resetWorkflowState(id: string): boolean {
  try {
    const raw = JSON.parse(localStorage.getItem(WF_KEY) ?? '{}');
    const all = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
    delete all[id];
    localStorage.setItem(WF_KEY, JSON.stringify(all));
    return true;
  } catch { return false; }
}
