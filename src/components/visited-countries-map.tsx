'use client';

import { useEffect, useMemo, useState } from 'react';
import { geoEqualEarth, geoPath } from 'd3-geo';
import { feature } from 'topojson-client';
import { visitedCountries, overseasExperiences } from '@/data/overseas-experiences';

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
  const [topo, setTopo] = useState<{ objects: { countries: unknown } } | null>(null);
  const [hovered, setHovered] = useState<{ x: number; y: number; name: string; flag: string; trips: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/geo/countries-110m.json')
      .then((r) => r.json())
      .then((d) => { if (!cancelled) setTopo(d); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

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
        fill: v ? CONTINENT_FILL[v.continent] : 'var(--muted)',
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
      <div className="aspect-[800/420] w-full rounded-md bg-muted/40 animate-pulse flex items-center justify-center">
        <span className="text-xs text-muted-foreground">지도를 불러오는 중…</span>
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="relative w-full">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto rounded-md bg-background border" role="img" aria-label="Visited countries world map">
          {view.sphere && <path d={view.sphere} fill="var(--muted)" fillOpacity="0.3" stroke="hsl(var(--border))" strokeWidth={0.5} />}
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
          {view.pins.map((p) => (
            <g key={p.iso} className="cursor-pointer">
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
            <span className="text-muted-foreground"> · {hovered.trips}회 방문</span>
          </div>
        )}
      </div>

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
