"use client"

import { useEffect, useMemo, useRef } from "react"
import type { LatLngBoundsExpression, Map as LeafletMap } from "leaflet"

const CAMPUS_MAP_BOUNDS = {
  north: 14.64807,
  east: 121.08418,
  south: 14.63155,
  west: 121.07088,
} as const

const LANDMARKS = [
  { id: "gate-2", name: "Gate 2", lat: 14.635825, lng: 121.074719 },
  { id: "gate-2-9", name: "Gate 2.9", lat: 14.640421, lng: 121.074759 },
  { id: "gate-3", name: "Gate 3", lat: 14.640681, lng: 121.075508 },
  { id: "gate-3-2", name: "Gate 3.2", lat: 14.640904, lng: 121.074872 },
  { id: "gate-3-5", name: "Gate 3.5", lat: 14.64119, lng: 121.07477 },
] as const

const LOS_BY_DATE: Record<string, Record<string, string>> = {
  "2026-04-13": {
    "gate-2": "C",
    "gate-2-9": "D",
    "gate-3": "E",
    "gate-3-2": "D",
    "gate-3-5": "C",
  },
  "2026-03-15": {
    "gate-2": "C",
    "gate-2-9": "D",
    "gate-3": "E",
    "gate-3-2": "D",
    "gate-3-5": "C",
  },
}

function resolveLos(dateKey: string | null | undefined, landmarkId: string) {
  if (!dateKey) {
    return "LOS —"
  }

  return LOS_BY_DATE[dateKey]?.[landmarkId] ?? "LOS —"
}

function formatDateLabel(dateKey: string | null | undefined) {
  if (!dateKey) {
    return "No date selected"
  }

  const parsed = new Date(`${dateKey}T00:00:00`)
  if (Number.isNaN(parsed.getTime())) {
    return dateKey
  }

  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  })
}

function getLosColor(los: string) {
  if (los.endsWith("A") || los.endsWith("B") || los.endsWith("C")) {
    return "#22C55E"
  }
  if (los.endsWith("D") || los.endsWith("E")) {
    return "#F97316"
  }
  if (los.endsWith("F")) {
    return "#EF4444"
  }
  return "#94A3B8"
}

interface CampusOsmMapProps {
  selectedDate?: string | null
  className?: string
}

export function CampusOsmMap({ selectedDate, className }: CampusOsmMapProps) {
  const mapHostRef = useRef<HTMLDivElement | null>(null)
  const dateLabel = useMemo(() => formatDateLabel(selectedDate), [selectedDate])

  useEffect(() => {
    if (!mapHostRef.current) {
      return
    }

    let map: LeafletMap | null = null
    let isCancelled = false

    const mapBounds: LatLngBoundsExpression = [
      [CAMPUS_MAP_BOUNDS.south, CAMPUS_MAP_BOUNDS.west],
      [CAMPUS_MAP_BOUNDS.north, CAMPUS_MAP_BOUNDS.east],
    ]

    void (async () => {
      const L = await import("leaflet")
      if (isCancelled || !mapHostRef.current) {
        return
      }

      map = L.map(mapHostRef.current, {
        minZoom: 16,
        maxZoom: 20,
        maxBounds: mapBounds,
        maxBoundsViscosity: 1,
        zoomControl: true,
        scrollWheelZoom: true,
      })

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 20,
        minZoom: 16,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map)

      const gate3Landmark = LANDMARKS.find((landmark) => landmark.id === "gate-3")
      if (gate3Landmark) {
        map.setView([gate3Landmark.lat, gate3Landmark.lng], 17)
      } else {
        map.fitBounds(mapBounds, { padding: [12, 12] })
      }

      LANDMARKS.forEach((landmark) => {
        const los = resolveLos(selectedDate, landmark.id)
        const markerColor = getLosColor(los)
        const marker = L.circleMarker([landmark.lat, landmark.lng], {
          radius: 8,
          color: "#E2E8F0",
          weight: 1.5,
          fillColor: markerColor,
          fillOpacity: 0.95,
        }).addTo(map)

        const tooltipContent = `${landmark.name} | ${los} | ${dateLabel}`
        marker.bindTooltip(tooltipContent, {
          direction: "top",
          offset: [0, -8],
          sticky: true,
        })

        const popupContent = `
          <div style="font-family: Inter, system-ui, sans-serif; min-width: 160px;">
            <div style="font-weight: 700; margin-bottom: 4px;">${landmark.name}</div>
            <div><strong>Date:</strong> ${dateLabel}</div>
            <div><strong>LOS:</strong> ${los}</div>
            <div style="margin-top: 4px; color: #475569; font-size: 12px;">${landmark.lat.toFixed(6)}, ${landmark.lng.toFixed(6)}</div>
          </div>
        `

        marker.bindPopup(popupContent)
        marker.on("mouseover", () => marker.openTooltip())
        marker.on("click", () => {
          if (map) {
            const targetZoom = Math.max(map.getZoom(), 18)
            map.flyTo([landmark.lat, landmark.lng], targetZoom, { duration: 0.45 })
          }
          marker.openPopup()
        })
      })
    })()

    return () => {
      isCancelled = true
      map?.remove()
    }
  }, [dateLabel, selectedDate])

  return (
    <div className={className ?? "h-72 w-full rounded-xl"}>
      <div ref={mapHostRef} className="h-full w-full" />
    </div>
  )
}
