"use client"

import { CampusOsmMap } from "@/components/maps/campus-osm-map"
import type { PTSIMapResponse } from "@/lib/api"

interface OcclusionMapProps {
  data?: PTSIMapResponse | null
  loading?: boolean
  selectedDate?: string
}

export function OcclusionMap({ data, loading = false, selectedDate }: OcclusionMapProps) {

  return (
    <div className="rounded-3xl border border-border bg-card p-6 shadow-elevated">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-foreground">OpenStreetMap LOS View</h3>
          <p className="text-sm text-muted-foreground">Live LOS markers aligned to the currently selected dashboard date and time window.</p>
        </div>
      </div>

      <div className="relative">
        <CampusOsmMap
          selectedDate={selectedDate}
          occlusionData={data}
          className="h-[clamp(16rem,42vh,30rem)] w-full rounded-2xl border border-border"
        />
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-background/50 text-sm text-muted-foreground backdrop-blur-[1px]">
            Loading map context...
          </div>
        )}
      </div>

      <div className="mt-4 rounded-2xl border border-border/60 bg-secondary/40 p-3 text-xs text-muted-foreground">
        Click or hover each marker to inspect LOS for the current page selection. Marker colors follow LOS severity from A (lowest) to F (worst).
      </div>
    </div>
  )
}
