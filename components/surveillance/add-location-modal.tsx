"use client"

import { useState, useEffect, useCallback } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { MapPin, Search, Loader2 } from "lucide-react"

interface AddLocationModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onAddLocation?: (data: {
    name: string
    latitude: number
    longitude: number
    description: string
    address: string
  }) => void
}

export function AddLocationModal({ open, onOpenChange, onAddLocation }: AddLocationModalProps) {
  const [name, setName] = useState("")
  const [latitude, setLatitude] = useState("")
  const [longitude, setLongitude] = useState("")
  const [description, setDescription] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const [address, setAddress] = useState("")
  const [isSearching, setIsSearching] = useState(false)

  // Simulated place search (in production, use Google Places API)
  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) return
    
    setIsSearching(true)
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 800))
    
    // Mock search results - in production, use Google Places Autocomplete
    const mockLocations: Record<string, { lat: number; lng: number; address: string }> = {
      "new york": { lat: 40.7128, lng: -74.0060, address: "New York, NY, USA" },
      "los angeles": { lat: 34.0522, lng: -118.2437, address: "Los Angeles, CA, USA" },
      "chicago": { lat: 41.8781, lng: -87.6298, address: "Chicago, IL, USA" },
      "manila": { lat: 14.5995, lng: 120.9842, address: "Manila, Philippines" },
      "tokyo": { lat: 35.6762, lng: 139.6503, address: "Tokyo, Japan" },
    }
    
    const query = searchQuery.toLowerCase()
    const found = Object.entries(mockLocations).find(([key]) => query.includes(key))
    
    if (found) {
      const [, coords] = found
      setLatitude(coords.lat.toString())
      setLongitude(coords.lng.toString())
      setAddress(coords.address)
    }
    
    setIsSearching(false)
  }, [searchQuery])

  const handleSubmit = () => {
    if (name && latitude && longitude) {
      onAddLocation?.({
        name,
        latitude: parseFloat(latitude),
        longitude: parseFloat(longitude),
        description,
        address,
      })
      handleClose()
    }
  }

  const handleClose = () => {
    setName("")
    setLatitude("")
    setLongitude("")
    setDescription("")
    setSearchQuery("")
    setAddress("")
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="bg-card border-border sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-foreground">
            <MapPin className="w-5 h-5" />
            Add New Location
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            Search for a place or enter GPS coordinates manually.
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          {/* Place Search */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Search Location</label>
            <div className="flex gap-2">
              <Input 
                placeholder="Search for a place..." 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
              <Button 
                variant="outline" 
                onClick={handleSearch}
                disabled={isSearching}
                className="border-border shrink-0"
              >
                {isSearching ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
              </Button>
            </div>
            {address && (
              <p className="text-xs text-muted-foreground">Found: {address}</p>
            )}
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Location Name</label>
            <Input 
              placeholder="e.g., South Entrance" 
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Latitude</label>
              <Input 
                type="number" 
                step="0.000001"
                placeholder="40.7128" 
                value={latitude}
                onChange={(e) => setLatitude(e.target.value)}
                className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Longitude</label>
              <Input 
                type="number" 
                step="0.000001"
                placeholder="-74.0060" 
                value={longitude}
                onChange={(e) => setLongitude(e.target.value)}
                className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
            </div>
          </div>

          {/* Mini Map Preview */}
          {latitude && longitude && (
            <div className="rounded-lg overflow-hidden border border-border bg-muted h-32 flex items-center justify-center">
              <div className="text-center">
                <MapPin className="w-6 h-6 text-primary mx-auto mb-1" />
                <p className="text-xs text-muted-foreground">
                  {parseFloat(latitude).toFixed(4)}, {parseFloat(longitude).toFixed(4)}
                </p>
              </div>
            </div>
          )}
          
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Description (Optional)</label>
            <Input 
              placeholder="Brief description of the location" 
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose} className="border-border text-foreground">
            Cancel
          </Button>
          <Button 
            onClick={handleSubmit}
            disabled={!name || !latitude || !longitude}
            className="bg-primary text-primary-foreground hover:bg-primary/90"
          >
            Add Location
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
