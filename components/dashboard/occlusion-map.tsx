"use client"

import { Clock } from "lucide-react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { CampusOsmMap } from "@/components/maps/campus-osm-map"
import type { PTSIMapResponse } from "@/lib/api"

interface OcclusionMapProps {
  hourFilter: string
  onHourFilterChange: (hour: string) => void
  data?: PTSIMapResponse | null
  loading?: boolean
  selectedDate?: string
}

function formatHourLabel(hour: string) {
  const [rawHours, rawMinutes] = hour.split(":")
  const hours = Number(rawHours)
  const minutes = Number(rawMinutes ?? 0)
  const suffix = hours >= 12 ? "PM" : "AM"
  const displayHours = ((hours + 11) % 12) + 1
  return `${displayHours}:${String(minutes).padStart(2, "0")} ${suffix}`
}

export function OcclusionMap({ hourFilter, onHourFilterChange, data, loading = false, selectedDate }: OcclusionMapProps) {
  const availableHours = Array.from(new Set(data?.availableHours ?? []))

  return (
    <div className="rounded-3xl border border-border bg-card p-6 shadow-elevated">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-foreground">OpenStreetMap LOS View</h3>
          <p className="text-sm text-muted-foreground">Leaflet map bounded to the campus rectangle with interactive gate landmarks.</p>
        </div>

        <Select value={hourFilter} onValueChange={onHourFilterChange}>
          <SelectTrigger className="w-44 rounded-2xl border-border bg-secondary text-foreground">
            <Clock className="mr-2 h-4 w-4 text-muted-foreground" />
            <SelectValue placeholder="Hour" />
          </SelectTrigger>
          <SelectContent className="rounded-xl border-border bg-popover">
            <SelectItem value="all" className="rounded-lg text-foreground">All Hours</SelectItem>
            {availableHours.map((hour) => (
              <SelectItem key={hour} value={hour} className="rounded-lg text-foreground">
                {formatHourLabel(hour)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="relative">
        <CampusOsmMap selectedDate={selectedDate} className="aspect-[21/9] min-h-[24rem] w-full rounded-2xl border border-border" />
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-background/50 text-sm text-muted-foreground backdrop-blur-[1px]">
            Loading map context...
          </div>
        )}
      </div>

      <div className="mt-4 rounded-2xl border border-border/60 bg-secondary/40 p-3 text-xs text-muted-foreground">
        Click or hover each marker to inspect LOS for the currently selected date. LOS values are currently mock values intended for frontend preview.
      </div>
    </div>
  )
}
