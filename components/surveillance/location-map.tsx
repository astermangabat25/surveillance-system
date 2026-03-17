"use client"

import { MapPin } from "lucide-react"

interface Location {
  id: string
  name: string
  latitude: number
  longitude: number
  videos: Array<{ id: string }>
}

interface LocationMapProps {
  locations: Location[]
}

export function LocationMap({ locations }: LocationMapProps) {
  // Calculate bounds for the map
  const minLat = Math.min(...locations.map(l => l.latitude))
  const maxLat = Math.max(...locations.map(l => l.latitude))
  const minLng = Math.min(...locations.map(l => l.longitude))
  const maxLng = Math.max(...locations.map(l => l.longitude))
  
  // Calculate normalized positions (0-100)
  const getPosition = (lat: number, lng: number) => {
    const latRange = maxLat - minLat || 0.01
    const lngRange = maxLng - minLng || 0.01
    
    return {
      x: ((lng - minLng) / lngRange) * 80 + 10, // 10-90% range
      y: ((maxLat - lat) / latRange) * 60 + 20, // 20-80% range (inverted for map)
    }
  }

  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-foreground">Location Overview</h3>
        <span className="text-xs text-muted-foreground">{locations.length} locations</span>
      </div>
      
      {/* Simple Map Visualization */}
      <div className="relative h-40 bg-muted/50 rounded-lg overflow-hidden">
        {/* Grid lines */}
        <div className="absolute inset-0">
          {[0, 25, 50, 75, 100].map((percent) => (
            <div
              key={`h-${percent}`}
              className="absolute w-full border-t border-border/30"
              style={{ top: `${percent}%` }}
            />
          ))}
          {[0, 25, 50, 75, 100].map((percent) => (
            <div
              key={`v-${percent}`}
              className="absolute h-full border-l border-border/30"
              style={{ left: `${percent}%` }}
            />
          ))}
        </div>
        
        {/* Location markers */}
        {locations.map((location) => {
          const pos = getPosition(location.latitude, location.longitude)
          return (
            <div
              key={location.id}
              className="absolute transform -translate-x-1/2 -translate-y-1/2 group cursor-pointer"
              style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
            >
              {/* Glow effect */}
              <div className="absolute inset-0 w-8 h-8 -ml-3 -mt-3 rounded-full bg-primary/20 animate-pulse" />
              
              {/* Marker */}
              <div className="relative w-4 h-4 rounded-full bg-primary border-2 border-white shadow-lg flex items-center justify-center">
                <div className="w-1.5 h-1.5 rounded-full bg-white" />
              </div>
              
              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-foreground text-background text-xs rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                {location.name}
                <span className="ml-1 text-muted-foreground">({location.videos.length} videos)</span>
              </div>
            </div>
          )
        })}
        
        {/* Map label */}
        <div className="absolute bottom-2 right-2 text-[10px] text-muted-foreground bg-card/80 px-2 py-1 rounded">
          Surveillance Area
        </div>
      </div>
    </div>
  )
}
