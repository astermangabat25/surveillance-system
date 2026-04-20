"use client"

import { useEffect, useMemo, useRef } from "react"
import type { LatLngBoundsExpression, Map as LeafletMap } from "leaflet"
import type { PTSIMapResponse, PTSILocation } from "@/lib/api"

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

interface CampusOsmMapProps {
  selectedDate?: string | null
  occlusionData?: PTSIMapResponse | null
  className?: string
}

const LOS_RANKS: Record<string, number> = {
  A: 0,
  B: 1,
  C: 2,
  D: 3,
  E: 4,
  F: 5,
}

function normalizeLosGrade(value: unknown): string | null {
  const los = String(value ?? "").trim().toUpperCase()
  return los in LOS_RANKS ? los : null
}

function resolveLosFromLocation(location: PTSILocation) {
  const directLos = normalizeLosGrade(location.los)
  if (directLos) {
    return directLos
  }

  const hourlyLosValues = location.hourlyScores
    .map((hourlyScore) => normalizeLosGrade(hourlyScore.los))
    .filter((los): los is string => los !== null)

  if (hourlyLosValues.length === 0) {
    return null
  }

  return hourlyLosValues.reduce((worst, current) =>
    LOS_RANKS[current] > LOS_RANKS[worst] ? current : worst,
  )
}

function squaredDistance(lat1: number, lng1: number, lat2: number, lng2: number) {
  const dLat = lat1 - lat2
  const dLng = lng1 - lng2
  return (dLat * dLat) + (dLng * dLng)
}

function colorForLos(los: string | null) {
  if (los === "A") return "#22C55E"
  if (los === "B") return "#84CC16"
  if (los === "C") return "#EAB308"
  if (los === "D") return "#F97316"
  if (los === "E") return "#EF4444"
  if (los === "F") return "#B91C1C"

  return "#94A3B8"
}

function normalizeLocationName(name: string) {
  return name.toLowerCase().replace(/[^a-z0-9.]+/g, "").trim()
}

export function CampusOsmMap({ selectedDate, occlusionData, className }: CampusOsmMapProps) {
  const mapHostRef = useRef<HTMLDivElement | null>(null)
  const dateLabel = useMemo(() => formatDateLabel(selectedDate), [selectedDate])

  const markers = useMemo(() => {
    const locationCandidates = occlusionData?.locations ?? []
    const locationByName = new Map(locationCandidates.map((location) => [normalizeLocationName(location.name), location]))
    const locationsWithValidCoords = locationCandidates.filter(
      (location) => Number.isFinite(location.latitude) && Number.isFinite(location.longitude),
    )

    return LANDMARKS.map((landmark) => {
      const byName = locationByName.get(normalizeLocationName(landmark.name))
      const nearestLocation = locationsWithValidCoords.reduce<PTSILocation | null>((closest, location) => {
        if (!closest) {
          return location
        }

        const currentDistance = squaredDistance(landmark.lat, landmark.lng, location.latitude, location.longitude)
        const closestDistance = squaredDistance(landmark.lat, landmark.lng, closest.latitude, closest.longitude)
        return currentDistance < closestDistance ? location : closest
      }, null)

      const matchedLocation = byName ?? nearestLocation
      const los = matchedLocation ? resolveLosFromLocation(matchedLocation) : null

      const markerLos = los ? `LOS ${los}` : "LOS -"
      const sourceLabel = byName
        ? `Synced from ${byName.name}`
        : nearestLocation
          ? `Synced (nearest) from ${nearestLocation.name}`
          : "No LOS data for current selection"

      return {
        id: landmark.id,
        name: landmark.name,
        lat: landmark.lat,
        lng: landmark.lng,
        los: markerLos,
        markerColor: colorForLos(los),
        sourceLabel,
      }
    })
  }, [occlusionData])

  useEffect(() => {
    if (!mapHostRef.current) {
      return
    }

    let map: LeafletMap | null = null
    let isCancelled = false
    let resizeObserver: ResizeObserver | null = null
    let delayedInvalidateTimer: number | null = null

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

      if (markers.length > 0) {
        const markerBounds: LatLngBoundsExpression = markers.map((marker) => [marker.lat, marker.lng])
        map.fitBounds(markerBounds, { padding: [24, 24], maxZoom: 18 })
      } else {
        map.fitBounds(mapBounds, { padding: [12, 12] })
      }

      const safelyInvalidateSize = () => {
        if (!map || isCancelled) {
          return
        }

        map.invalidateSize()
      }

      requestAnimationFrame(safelyInvalidateSize)
      delayedInvalidateTimer = window.setTimeout(safelyInvalidateSize, 180)

      resizeObserver = new ResizeObserver(() => {
        safelyInvalidateSize()
      })
      resizeObserver.observe(mapHostRef.current)

      markers.forEach((markerData) => {
        const marker = L.circleMarker([markerData.lat, markerData.lng], {
          radius: 8,
          color: "#E2E8F0",
          weight: 1.5,
          fillColor: markerData.markerColor,
          fillOpacity: 0.95,
        }).addTo(map)

        const tooltipContent = `${markerData.name} | ${markerData.los} | ${dateLabel}`
        marker.bindTooltip(tooltipContent, {
          direction: "top",
          offset: [0, -8],
          sticky: true,
        })

        const popupContent = `
          <div style="font-family: Inter, system-ui, sans-serif; min-width: 160px;">
            <div style="font-weight: 700; margin-bottom: 4px;">${markerData.name}</div>
            <div><strong>Date:</strong> ${dateLabel}</div>
            <div><strong>LOS:</strong> ${markerData.los}</div>
            <div><strong>Source:</strong> ${markerData.sourceLabel}</div>
            <div style="margin-top: 4px; color: #475569; font-size: 12px;">${markerData.lat.toFixed(6)}, ${markerData.lng.toFixed(6)}</div>
          </div>
        `

        marker.bindPopup(popupContent)
        marker.on("mouseover", () => marker.openTooltip())
        marker.on("click", () => {
          if (map) {
            const targetZoom = Math.max(map.getZoom(), 18)
            map.flyTo([markerData.lat, markerData.lng], targetZoom, { duration: 0.45 })
          }
          marker.openPopup()
        })
      })
    })()

    return () => {
      isCancelled = true
      if (delayedInvalidateTimer !== null) {
        window.clearTimeout(delayedInvalidateTimer)
      }
      resizeObserver?.disconnect()
      map?.remove()
    }
  }, [dateLabel, markers])

  return (
    <div className={`${className ?? "h-72 w-full rounded-xl"} relative overflow-hidden`}>
      <div ref={mapHostRef} className="h-full w-full" />
    </div>
  )
}
