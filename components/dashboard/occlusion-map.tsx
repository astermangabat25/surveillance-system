"use client"

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Clock, MapPin } from "lucide-react"

interface OcclusionMapProps {
  hourFilter: string
  onHourFilterChange: (hour: string) => void
}

// Mock location data with occlusion severity
const locations = [
  { id: "north-gate", name: "North Gate", lat: 14.5547, lng: 121.0244, severity: "heavy" },
  { id: "main-hall", name: "Main Hall", lat: 14.5565, lng: 121.0220, severity: "moderate" },
  { id: "parking-a", name: "Parking Lot A", lat: 14.5530, lng: 121.0260, severity: "light" },
  { id: "south-entrance", name: "South Entrance", lat: 14.5510, lng: 121.0235, severity: "moderate" },
  { id: "east-wing", name: "East Wing", lat: 14.5555, lng: 121.0280, severity: "heavy" },
]

// Severity levels vary by hour for demo
const severityByHour: Record<string, Record<string, "light" | "moderate" | "heavy">> = {
  "all": { "north-gate": "heavy", "main-hall": "moderate", "parking-a": "light", "south-entrance": "moderate", "east-wing": "heavy" },
  "6": { "north-gate": "light", "main-hall": "light", "parking-a": "light", "south-entrance": "light", "east-wing": "light" },
  "8": { "north-gate": "moderate", "main-hall": "light", "parking-a": "light", "south-entrance": "moderate", "east-wing": "light" },
  "10": { "north-gate": "heavy", "main-hall": "moderate", "parking-a": "light", "south-entrance": "moderate", "east-wing": "moderate" },
  "12": { "north-gate": "heavy", "main-hall": "heavy", "parking-a": "moderate", "south-entrance": "heavy", "east-wing": "moderate" },
  "14": { "north-gate": "heavy", "main-hall": "heavy", "parking-a": "heavy", "south-entrance": "heavy", "east-wing": "heavy" },
  "16": { "north-gate": "moderate", "main-hall": "heavy", "parking-a": "moderate", "south-entrance": "moderate", "east-wing": "heavy" },
  "18": { "north-gate": "moderate", "main-hall": "moderate", "parking-a": "light", "south-entrance": "moderate", "east-wing": "moderate" },
  "20": { "north-gate": "light", "main-hall": "light", "parking-a": "light", "south-entrance": "light", "east-wing": "light" },
}

const getSeverityColor = (severity: string) => {
  switch (severity) {
    case "heavy": return { bg: "#EF4444", glow: "rgba(239, 68, 68, 0.6)", shadow: "0 0 30px 15px rgba(239, 68, 68, 0.4)" }
    case "moderate": return { bg: "#F97316", glow: "rgba(249, 115, 22, 0.5)", shadow: "0 0 24px 12px rgba(249, 115, 22, 0.35)" }
    case "light": return { bg: "#EAB308", glow: "rgba(234, 179, 8, 0.4)", shadow: "0 0 18px 9px rgba(234, 179, 8, 0.3)" }
    default: return { bg: "#22C55E", glow: "rgba(34, 197, 94, 0.3)", shadow: "0 0 12px 6px rgba(34, 197, 94, 0.2)" }
  }
}

export function OcclusionMap({ hourFilter, onHourFilterChange }: OcclusionMapProps) {
  const currentSeverities = severityByHour[hourFilter] || severityByHour["all"]
  
  // Calculate bounds for the map
  const minLat = Math.min(...locations.map(l => l.lat))
  const maxLat = Math.max(...locations.map(l => l.lat))
  const minLng = Math.min(...locations.map(l => l.lng))
  const maxLng = Math.max(...locations.map(l => l.lng))
  
  const getPosition = (lat: number, lng: number) => {
    const latRange = maxLat - minLat || 0.01
    const lngRange = maxLng - minLng || 0.01
    
    return {
      x: ((lng - minLng) / lngRange) * 70 + 15,
      y: ((maxLat - lat) / latRange) * 60 + 20,
    }
  }

  return (
    <div className="rounded-xl border border-border bg-card p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-semibold text-foreground">Occlusion Severity Map</h3>
          <p className="text-sm text-muted-foreground">Pedestrian detection difficulty by location</p>
        </div>
        
        {/* Hour Filter */}
        <Select value={hourFilter} onValueChange={onHourFilterChange}>
          <SelectTrigger className="w-40 bg-secondary border-border text-foreground">
            <Clock className="w-4 h-4 mr-2 text-muted-foreground" />
            <SelectValue placeholder="Hour" />
          </SelectTrigger>
          <SelectContent className="bg-card border-border">
            <SelectItem value="all" className="text-foreground">All Hours</SelectItem>
            <SelectItem value="6" className="text-foreground">6:00 AM</SelectItem>
            <SelectItem value="8" className="text-foreground">8:00 AM</SelectItem>
            <SelectItem value="10" className="text-foreground">10:00 AM</SelectItem>
            <SelectItem value="12" className="text-foreground">12:00 PM</SelectItem>
            <SelectItem value="14" className="text-foreground">2:00 PM</SelectItem>
            <SelectItem value="16" className="text-foreground">4:00 PM</SelectItem>
            <SelectItem value="18" className="text-foreground">6:00 PM</SelectItem>
            <SelectItem value="20" className="text-foreground">8:00 PM</SelectItem>
          </SelectContent>
        </Select>
      </div>
      
      {/* Map Visualization */}
      <div className="relative h-64 bg-slate-900 rounded-lg overflow-hidden">
        {/* Grid lines */}
        <div className="absolute inset-0">
          {[0, 25, 50, 75, 100].map((percent) => (
            <div
              key={`h-${percent}`}
              className="absolute w-full border-t border-slate-700/30"
              style={{ top: `${percent}%` }}
            />
          ))}
          {[0, 25, 50, 75, 100].map((percent) => (
            <div
              key={`v-${percent}`}
              className="absolute h-full border-l border-slate-700/30"
              style={{ left: `${percent}%` }}
            />
          ))}
        </div>
        
        {/* Location markers with glow */}
        {locations.map((location) => {
          const pos = getPosition(location.lat, location.lng)
          const severity = currentSeverities[location.id] || "light"
          const colors = getSeverityColor(severity)
          
          return (
            <div
              key={location.id}
              className="absolute transform -translate-x-1/2 -translate-y-1/2 group cursor-pointer"
              style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
            >
              {/* Outer glow - larger for higher severity */}
              <div 
                className="absolute rounded-full animate-pulse"
                style={{ 
                  width: severity === "heavy" ? "60px" : severity === "moderate" ? "48px" : "36px",
                  height: severity === "heavy" ? "60px" : severity === "moderate" ? "48px" : "36px",
                  left: "50%",
                  top: "50%",
                  transform: "translate(-50%, -50%)",
                  background: `radial-gradient(circle, ${colors.glow} 0%, transparent 70%)`,
                  boxShadow: colors.shadow,
                }}
              />
              
              {/* Inner marker */}
              <div 
                className="relative w-4 h-4 rounded-full border-2 border-white shadow-lg"
                style={{ backgroundColor: colors.bg }}
              >
                <div className="absolute inset-0 rounded-full bg-white/30" />
              </div>
              
              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-3 px-3 py-2 bg-slate-800 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-xl border border-slate-700 z-10">
                <p className="font-medium">{location.name}</p>
                <p className="text-slate-400 capitalize">Occlusion: {severity}</p>
              </div>
            </div>
          )
        })}
        
        {/* Map label */}
        <div className="absolute bottom-3 right-3 text-[10px] text-slate-400 bg-slate-800/80 px-2 py-1 rounded border border-slate-700">
          <MapPin className="w-3 h-3 inline mr-1" />
          Surveillance Zone
        </div>
      </div>
      
      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-4 pt-4 border-t border-border">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <span className="text-xs text-muted-foreground">Light Occlusion</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-orange-500" />
          <span className="text-xs text-muted-foreground">Moderate Occlusion</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span className="text-xs text-muted-foreground">Heavy Occlusion</span>
        </div>
      </div>
    </div>
  )
}
