"use client"

import { MapPin } from "lucide-react"
import { CampusOsmMap } from "@/components/maps/campus-osm-map"

interface Location {
  id: string
  name: string
  address?: string
  latitude: number
  longitude: number
  videos: Array<{ id: string }>
}

interface LocationMapProps {
  locations: Location[]
  selectedDate?: string
}

export function LocationMap({ locations, selectedDate }: LocationMapProps) {
  return (
    <div className="rounded-2xl border border-border bg-secondary/50 p-3 shadow-elevated-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Campus Landmark Map</h3>
        <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">{locations.length} locations</span>
      </div>

      <CampusOsmMap selectedDate={selectedDate} className="aspect-[16/9] min-h-[21rem] w-full rounded-xl border border-border" />

      <p className="mt-3 text-[11px] text-muted-foreground">
        Hover or click Gate 2, Gate 2.9, Gate 3, Gate 3.2, and Gate 3.5 to view LOS for the selected date.
      </p>

      <div className="mt-3 space-y-2">
        {locations.map((location) => (
          <div key={location.id} className="flex items-center justify-between rounded-xl border border-border bg-background/40 px-3 py-2">
            <div className="min-w-0">
              <p className="truncate text-xs font-medium text-foreground">{location.name}</p>
              <p className="truncate text-[11px] text-muted-foreground">{location.address}</p>
            </div>
            <span className="ml-3 inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
              <MapPin className="h-2.5 w-2.5" />
              {location.videos.length} feeds
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
