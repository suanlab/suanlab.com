import { isCalendarDate } from './content-date';

export interface DeadlineInfo {
  type: string;
  date: string | null;
  note?: string;
  kind?: 'submission' | 'abstract' | 'event' | 'milestone';
  status?: 'tentative';
  time?: string;
  timezone?: string | null;
  endDate?: string;
}
interface ScheduleInfo {
  timezone?: string;
  deadlineTime?: string;
  deadlines: DeadlineInfo[];
}
export type DeadlineScope = 'submission' | 'schedule';

export function deadlineKind(d: DeadlineInfo): NonNullable<DeadlineInfo['kind']> {
  if (d.kind) return d.kind;
  if (/conference/i.test(d.type)) return 'event';
  if (/rebuttal|response|review|notification|decision|camera|supplement|video|cancellation|submission opens/i.test(d.type)) return 'milestone';
  if (/abstract|paper registration/i.test(d.type)) return 'abstract';
  if (/tutorial|satellite|proposal|registration/i.test(d.type)) return 'milestone';
  return /paper|submission|commitment|show.*tell/i.test(d.type) ? 'submission' : 'milestone';
}

/** Exact instants require a published timezone; conference days remain date-only. */
export function deadlineInstant(d: DeadlineInfo, conf: ScheduleInfo): Date | null {
  const timezone = d.timezone === undefined ? conf.timezone : d.timezone;
  if (!d.date || !isCalendarDate(d.date) || d.status === 'tentative' || deadlineKind(d) === 'event' || !timezone) return null;
  const time = d.time ?? conf.deadlineTime ?? (['AoE', 'UTC'].includes(timezone) ? '23:59:59' : '');
  if (!/^([01]\d|2[0-3]):[0-5]\d:[0-5]\d$/.test(time)) return null;
  if (timezone === 'AoE' || timezone === 'UTC') return new Date(`${d.date}T${time}${timezone === 'AoE' ? '-12:00' : 'Z'}`);
  try {
    const formatter = new Intl.DateTimeFormat('en-CA', { timeZone: timezone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' });
    const local = Date.parse(`${d.date}T${time}Z`);
    let utc = local;
    for (let i = 0; i < 3; i++) {
      const p = Object.fromEntries(formatter.formatToParts(new Date(utc)).map(part => [part.type, part.value]));
      const represented = Date.parse(`${p.year}-${p.month}-${p.day}T${p.hour}:${p.minute}:${p.second}Z`);
      utc += local - represented;
    }
    return new Date(utc);
  } catch { return null; }
}

export function isUpcoming(d: DeadlineInfo, conf: ScheduleInfo, now: Date): boolean {
  if (!d.date || !isCalendarDate(d.date) || d.status === 'tentative') return false;
  const instant = deadlineInstant(d, conf);
  return instant ? instant.getTime() > now.getTime() : (d.endDate ?? d.date) >= now.toISOString().slice(0, 10);
}

export function upcomingDeadlines<T extends DeadlineInfo>(conf: Omit<ScheduleInfo, 'deadlines'> & { deadlines: T[] }, now: Date, scope: DeadlineScope): T[] {
  return conf.deadlines.filter(d => isUpcoming(d, conf, now) && (scope === 'schedule' || ['submission', 'abstract'].includes(deadlineKind(d))))
    .sort((a, b) => (deadlineInstant(a, conf)?.getTime() ?? Date.parse(`${a.date}T23:59:59Z`)) - (deadlineInstant(b, conf)?.getTime() ?? Date.parse(`${b.date}T23:59:59Z`)));
}

export function submissionStatus(conf: ScheduleInfo, now: Date): 'upcoming' | 'closed' | 'tba' {
  if (upcomingDeadlines(conf, now, 'submission').length) return 'upcoming';
  const submissions = conf.deadlines.filter(d => ['submission', 'abstract'].includes(deadlineKind(d)));
  if (submissions.some(d => !d.date || d.status === 'tentative') || !submissions.length) return 'tba';
  return 'closed';
}

function escapeText(text: string): string {
  return text.replace(/\\/g, '\\\\').replace(/\r?\n/g, '\\n').replace(/,/g, '\\,').replace(/;/g, '\\;').replace(/\r/g, '');
}
function foldLine(line: string): string {
  const encoder = new TextEncoder();
  let result = '', bytes = 0;
  for (const char of line) {
    const size = encoder.encode(char).length;
    if (bytes + size > 75) { result += '\r\n '; bytes = 1; }
    result += char; bytes += size;
  }
  return result;
}
const utcStamp = (date: Date) => date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');

/** RFC 5545: omit unconfirmed dates, convert exact deadlines to UTC, fold UTF-8 lines. */
export function conferenceCalendar(conferences: (ScheduleInfo & { id: string; name: string; year: number; url: string; location: string })[], now: Date, scope: DeadlineScope): string {
  const lines = ['BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//SuanLab//Research Deadlines//EN', 'CALSCALE:GREGORIAN'];
  for (const conf of conferences) for (const d of upcomingDeadlines(conf, now, scope)) {
    const instant = deadlineInstant(d, conf);
    const uid = `${conf.id}-${encodeURIComponent(d.type)}-${d.date}@suanlab.com`;
    lines.push('BEGIN:VEVENT', `UID:${uid}`, `DTSTAMP:${utcStamp(now)}`);
    if (instant) lines.push(`DTSTART:${utcStamp(instant)}`, `DTEND:${utcStamp(new Date(instant.getTime() + 60_000))}`);
    else {
      const end = new Date(`${d.endDate ?? d.date}T00:00:00Z`); end.setUTCDate(end.getUTCDate() + 1);
      lines.push(`DTSTART;VALUE=DATE:${d.date!.replace(/-/g, '')}`, `DTEND;VALUE=DATE:${end.toISOString().slice(0, 10).replace(/-/g, '')}`);
    }
    lines.push(`SUMMARY:${escapeText(`${conf.name} ${conf.year} — ${d.type}`)}`, `DESCRIPTION:${escapeText(`${d.note ?? ''}\n${instant ? d.timezone ?? conf.timezone : 'Date only; consult official source for time'}\n${conf.url}`)}`, `LOCATION:${escapeText(conf.location)}`, `URL:${conf.url}`, 'END:VEVENT');
  }
  lines.push('END:VCALENDAR');
  return lines.map(foldLine).join('\r\n') + '\r\n';
}
