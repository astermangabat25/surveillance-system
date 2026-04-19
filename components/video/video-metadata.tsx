"use client"

import { Calendar, Clock, MapPin, Users } from "lucide-react"

interface VideoMetadataProps {
  date: string
  startTime: string
  endTime: string
  gpsLat: number
  gpsLng: number
  trackedVehiclesSoFar: number
  vehicleCount: number
  currentLOS?: string | null
  worstLOS?: string | null
  averageLOS?: string | null
}

export function VideoMetadata({ 
  date, 
  startTime, 
  endTime, 
  gpsLat, 
  gpsLng,
  trackedVehiclesSoFar,
  vehicleCount,
  currentLOS,
  worstLOS,
  averageLOS,
}: VideoMetadataProps) {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
      <MetadataCard
        icon={Calendar}
        label="Date"
        value={date}
      />
      <MetadataCard
        icon={Clock}
        label="Time Range"
        value={`${startTime} - ${endTime}`}
      />
      <MetadataCard
        icon={MapPin}
        label="GPS Location"
        value={`${gpsLat.toFixed(4)}, ${gpsLng.toFixed(4)}`}
      />
      <MetadataCard
        icon={Users}
        label="Tracked Vehicles"
        value={trackedVehiclesSoFar.toString()}
        highlight
      />
      <MetadataCard
        icon={Users}
        label="Total Vehicles"
        value={vehicleCount.toString()}
        highlight
      />
      <MetadataCard
        icon={Users}
        label="LOS (Current/Worst)"
        value={`${currentLOS ?? "--"} / ${worstLOS ?? "--"}`}
        highlight
      />
      <MetadataCard
        icon={Users}
        label="LOS (Average)"
        value={averageLOS ?? "--"}
        highlight
      />
    </div>
  )
}

function MetadataCard({ 
  icon: Icon, 
  label, 
  value, 
  highlight = false 
}: { 
  icon: React.ElementType
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <div className="p-4 rounded-lg bg-card border border-border">
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`w-4 h-4 ${highlight ? 'text-primary' : 'text-muted-foreground'}`} />
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <p className={`text-sm font-medium ${highlight ? 'text-primary' : 'text-foreground'}`}>
        {value}
      </p>
    </div>
  )
}
