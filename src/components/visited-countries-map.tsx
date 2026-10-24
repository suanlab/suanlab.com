'use client';

import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { geoEqualEarth, geoPath } from 'd3-geo';
import { feature } from 'topojson-client';
import { visitedCountries, overseasExperiences } from '@/data/overseas-experiences';
import { useLanguage } from '@/components/language-provider';
import { useSceneActivity } from '@/components/motion/use-scene-activity';

const W = 800;
const H = 420;

// 대륙별 핀/채우기 색 (SVG)
const CONTINENT_FILL: Record<string, string> = {
  Asia: '#ef4444',
  Europe: '#3b82f6',
  America: '#22c55e',
  Oceania: '#a855f7',
  Africa: '#eab308',
};

interface FeatProps {
  name?: string;
}
interface GeoFeature {
  type: 'Feature';
  id?: string;
  properties: FeatProps;
  geometry: unknown;
}

export function VisitedCountriesMap() {
  const { language } = useLanguage();
  const tripInput = useId();
  const sceneRef = useRef<HTMLDivElement>(null);
  const running = useSceneActivity(sceneRef);
  const [tripId, setTripId] = useState(overseasExperiences[0]?.id ?? 0);
  const trip = overseasExperiences.find(item => item.id === tripId);
  const [loadError, setLoadError] = useState(false);
  const [loadVersion, setLoadVersion] = useState(0);
  const [topo, setTopo] = useState<{ objects: { countries: unknown } } | null>(null);
  const [hovered, setHovered] = useState<{ x: number; y: number; name: string; flag: string; trips: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10_000);
    setLoadError(false);
    fetch('/geo/countries-110m.json', { signal: controller.signal })
      .then(r => { if (!r.ok) throw new Error('Map download failed'); return r.json(); })
      .then(d => { if (!d?.objects?.countries) throw new Error('Invalid map data'); if (!cancelled) setTopo(d); })
      .catch(() => { if (!cancelled) setLoadError(true); })
      .finally(() => clearTimeout(timer));
    return () => { cancelled = true; clearTimeout(timer); controller.abort(); };
  }, [loadVersion]);

  const view = useMemo(() => {
    if (!topo) return null;
    const fc = feature(topo as never, topo.objects.countries as never) as unknown as { features: GeoFeature[] };
    const projection = geoEqualEarth().fitExtent([[8, 8], [W - 8, H - 8]], { type: 'Sphere' } as never);
    const path = geoPath(projection);

    const visitedByIso = new Map(visitedCountries.map((c) => [c.iso, c]));
    const sphere = path({ type: 'Sphere' } as never);

    const paths = fc.features.map((f) => {
      const v = f.id ? visitedByIso.get(f.id) : undefined;
      return {
        id: f.id ?? f.properties?.name,
        d: path(f as never),
        fill: v ? CONTINENT_FILL[v.continent] : 'hsl(var(--muted))',
        visited: !!v,
        name: f.properties?.name ?? f.id,
      };
    });

    const pins = visitedCountries
      .map((c) => {
        let xy: [number, number] | null = null;
        if (typeof c.lat === 'number' && typeof c.lng === 'number') {
          const p = projection([c.lng, c.lat] as never) as [number, number] | null;
          if (p) xy = p;
        } else {
          const f = fc.features.find((ff) => ff.id === c.iso);
          if (f) {
            const cen = path.centroid(f as never);
            if (cen && !Number.isNaN(cen[0])) xy = cen as [number, number];
          }
        }
        if (!xy) return null;
        const trips = overseasExperiences.filter((e) => e.countries.includes(c.name)).length;
        return { iso: c.iso, name: c.name, flag: c.flag, continent: c.continent, x: xy[0], y: xy[1], trips };
      })
      .filter((p): p is NonNullable<typeof p> => p !== null);

    return { sphere, paths, pins };
  }, [topo]);

  if (!view) {
    return (
      <div ref={sceneRef} role="status" className="aspect-[800/420] w-full rounded-md bg-muted/40 flex flex-col gap-3 items-center justify-center">
        <span className="text-xs text-foreground">{loadError
          ? (language === 'ko' ? '지도를 불러오지 못했습니다.' : 'Could not load the map.')
          : (language === 'ko' ? '지도를 불러오는 중…' : 'Loading map…')}</span>
        {loadError && <button type="button" className="text-sm text-primary underline" onClick={() => setLoadVersion(value => value + 1)}>{language === 'ko' ? '다시 시도' : 'Retry map'}</button>}
      </div>
    );
  }

  return (
    <div ref={sceneRef} data-motion={running ? 'running' : 'paused'} className="visited-map w-full">
      <label htmlFor={tripInput} className="mb-3 block text-xs font-medium">{language === 'ko' ? '여행·학회 경로' : 'Trip / conference route'}
        <select id={tripInput} value={tripId} onChange={event => setTripId(Number(event.target.value))} className="mt-2 block w-full min-w-0 rounded-lg border bg-background p-2 text-xs">
          {overseasExperiences.map(item => <option key={item.id} value={item.id}>{item.period} · {item.purpose}</option>)}
        </select>
      </label>
      <div className="relative w-full">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto rounded-md bg-background border" role="img" aria-label={language === 'ko' ? '방문 국가 세계 지도' : 'Visited countries world map'}>
          <desc>{trip ? `${trip.period} · ${trip.purpose} · ${trip.countries.join(', ')}` : ''}</desc>
          {view.sphere && <path d={view.sphere} fill="hsl(var(--muted))" fillOpacity="0.3" stroke="hsl(var(--border))" strokeWidth={0.5} />}
          {view.paths.map((p) => (
            <path
              key={p.id}
              d={p.d ?? ''}
              fill={p.fill}
              fillOpacity={p.visited ? 0.85 : 1}
              stroke="hsl(var(--background))"
              strokeWidth={0.4}
            />
          ))}
          <g key={tripId} data-motion-reveal="route" aria-hidden="true">
            {trip?.countries.slice(1).map((country, index) => {
              const from = view.pins.find(pin => pin.name === trip.countries[index]);
              const to = view.pins.find(pin => pin.name === country);
              if (!from || !to || Math.abs(to.x - from.x) > W * .55) return null;
              const d = `M${from.x},${from.y} Q${(from.x + to.x) / 2},${Math.max(12, Math.min(from.y, to.y) - Math.min(65, Math.hypot(to.x - from.x, to.y - from.y) * .4))} ${to.x},${to.y}`;
              return <g key={`${from.iso}-${to.iso}`}>
                <path data-motion-path className="map-route" d={d} pathLength="1" fill="none" stroke="hsl(var(--foreground))" strokeWidth="1.8" strokeOpacity=".6" />
                <path className="map-route-signal" d={d} pathLength="1" fill="none" stroke="hsl(var(--primary))" strokeWidth="3" strokeDasharray=".12 .88" strokeLinecap="round" />
              </g>;
            })}
          </g>
          {view.pins.map((p) => (
            <g key={p.iso} className="cursor-pointer">
              {trip?.countries.includes(p.name) && <circle className="map-pin-halo" cx={p.x} cy={p.y} r="8" fill="none" stroke="hsl(var(--foreground))" strokeWidth="1.5" />}
              <circle
                cx={p.x}
                cy={p.y}
                r={4.5}
                fill={CONTINENT_FILL[p.continent]}
                stroke="white"
                strokeWidth={1.2}
                onMouseEnter={() => setHovered({ x: p.x, y: p.y, name: p.name, flag: p.flag, trips: p.trips })}
                onMouseLeave={() => setHovered(null)}
              />
            </g>
          ))}
        </svg>

        {hovered && (
          <div
            className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-md border bg-popover px-2 py-1 text-xs shadow-md whitespace-nowrap"
            style={{ left: `${(hovered.x / W) * 100}%`, top: `${(hovered.y / H) * 100}%`, marginTop: -8 }}
          >
            <span className="mr-1">{hovered.flag}</span>
            <span className="font-medium">{hovered.name}</span>
            <span className="text-muted-foreground"> · {hovered.trips}{language === 'ko' ? '회 방문' : ' visits'}</span>
          </div>
        )}
      </div>

      <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground">{language === 'ko'
        ? `${trip?.countries.join(' → ') ?? ''} · 국가 위치를 잇는 개략 경로입니다.`
        : `${trip?.countries.join(' → ') ?? ''} · Schematic connections between country locations.`}</p>

      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
        {Object.entries(CONTINENT_FILL).map(([c, color]) => (
          <span key={c} className="inline-flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
            {c}
          </span>
        ))}
      </div>
    </div>
  );
}
