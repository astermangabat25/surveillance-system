"use client"

import { useState } from "react"
import { KPICards } from "@/components/dashboard/kpi-cards"
import { PedestrianChart } from "@/components/dashboard/pedestrian-chart"
import { OcclusionMap } from "@/components/dashboard/occlusion-map"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Calendar, Clock, Download, RefreshCw } from "lucide-react"

export default function DashboardPage() {
  const [selectedDate, setSelectedDate] = useState("2026-03-15")
  const [timeRange, setTimeRange] = useState("whole-day")
  const [hourFilter, setHourFilter] = useState("all")

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-border bg-card">
        <div>
          <h1 className="text-xl font-semibold text-foreground">System Analytics Dashboard</h1>
          <p className="text-sm text-muted-foreground">Real-time pedestrian tracking metrics</p>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Date Filter */}
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary border border-border">
            <Calendar className="w-4 h-4 text-muted-foreground" />
            <Input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="border-0 bg-transparent p-0 h-auto text-sm w-36 focus-visible:ring-0"
            />
          </div>

          {/* Time Range Filter */}
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-44 bg-secondary border-border text-foreground">
              <Clock className="w-4 h-4 mr-2 text-muted-foreground" />
              <SelectValue placeholder="Select time range" />
            </SelectTrigger>
            <SelectContent className="bg-card border-border">
              <SelectItem value="whole-day" className="text-foreground">Whole Day</SelectItem>
              <SelectItem value="last-1h" className="text-foreground">Last 1 Hour</SelectItem>
              <SelectItem value="last-3h" className="text-foreground">Last 3 Hours</SelectItem>
              <SelectItem value="last-6h" className="text-foreground">Last 6 Hours</SelectItem>
              <SelectItem value="last-12h" className="text-foreground">Last 12 Hours</SelectItem>
              <SelectItem value="morning" className="text-foreground">Morning (6AM-12PM)</SelectItem>
              <SelectItem value="afternoon" className="text-foreground">Afternoon (12PM-6PM)</SelectItem>
              <SelectItem value="evening" className="text-foreground">Evening (6PM-12AM)</SelectItem>
            </SelectContent>
          </Select>

          <Button variant="outline" className="border-border text-foreground hover:bg-secondary">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>

          <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
            <Download className="w-4 h-4 mr-2" />
            Export Report
          </Button>
        </div>
      </header>

      {/* Dashboard Content */}
      <div className="flex-1 overflow-auto p-6 space-y-6">
        {/* KPI Cards */}
        <KPICards />

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Main Chart */}
          <PedestrianChart timeRange={timeRange} selectedDate={selectedDate} />
          
          {/* Occlusion Severity Map */}
          <OcclusionMap hourFilter={hourFilter} onHourFilterChange={setHourFilter} />
        </div>
      </div>
    </div>
  )
}
