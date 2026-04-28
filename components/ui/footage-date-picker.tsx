"use client"

import { useEffect, useMemo, useState } from "react"
import { Calendar as CalendarIcon, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"

interface FootageDatePickerProps {
  value: string
  onChange: (value: string) => void
  highlightedDates?: string[]
  placeholder?: string
  allowClear?: boolean
  disabled?: boolean
  className?: string
}

function parseDateValue(value: string): Date | undefined {
  const parts = value.split("-").map((part) => Number(part))
  if (parts.length !== 3 || parts.some((part) => Number.isNaN(part))) {
    return undefined
  }

  const [year, month, day] = parts
  return new Date(year, month - 1, day)
}

function formatDateValue(date: Date): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  return `${year}-${month}-${day}`
}

function formatDateLabel(value: string, placeholder: string): string {
  const parsed = parseDateValue(value)
  if (!parsed) {
    return placeholder
  }

  return parsed.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  })
}

export function FootageDatePicker({
  value,
  onChange,
  highlightedDates = [],
  placeholder = "All dates",
  allowClear = false,
  disabled = false,
  className,
}: FootageDatePickerProps) {
  const [open, setOpen] = useState(false)
  const selectedDate = useMemo(() => parseDateValue(value), [value])
  const footageDates = useMemo(
    () => Array.from(new Set(highlightedDates)).map(parseDateValue).filter((date): date is Date => Boolean(date)),
    [highlightedDates],
  )
  const [visibleMonth, setVisibleMonth] = useState<Date | undefined>(selectedDate ?? footageDates[0])

  useEffect(() => {
    if (selectedDate) {
      setVisibleMonth(selectedDate)
    }
  }, [selectedDate])

  useEffect(() => {
    if (!selectedDate && footageDates.length > 0) {
      setVisibleMonth((current) => current ?? footageDates[0])
    }
  }, [footageDates, selectedDate])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <div
        className={cn(
          "box-border flex h-11 min-h-11 items-center gap-2 rounded-2xl border border-border bg-secondary px-4 shadow-xs transition-colors dark:bg-input/30 dark:hover:bg-input/50",
          className,
        )}
      >
        <CalendarIcon className="h-4 w-4 text-muted-foreground" />
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            disabled={disabled}
            className="h-full w-[152px] justify-start bg-transparent p-0 text-left text-sm font-normal hover:bg-transparent focus-visible:ring-0 dark:hover:bg-transparent"
          >
            <span className={value ? "text-foreground" : "text-muted-foreground"}>{formatDateLabel(value, placeholder)}</span>
          </Button>
        </PopoverTrigger>
        {allowClear && value ? (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-5 w-5 rounded-full hover:bg-muted"
            onClick={() => onChange("")}
          >
            <X className="h-3 w-3" />
          </Button>
        ) : null}
      </div>

      <PopoverContent align="end" className="w-auto overflow-hidden rounded-2xl border-border p-0">
        <Calendar
          mode="single"
          month={visibleMonth}
          onMonthChange={setVisibleMonth}
          selected={selectedDate}
          onSelect={(date) => {
            if (!date) {
              return
            }
            onChange(formatDateValue(date))
            setOpen(false)
          }}
          modifiers={footageDates.length > 0 ? { hasFootage: footageDates } : undefined}
        />
        <div className="flex items-center gap-2 border-t border-border px-4 py-2 text-xs text-muted-foreground">
          <span className="h-3 w-3 rounded-full ring-1 ring-inset ring-primary/70" />
          Days with uploaded footage
        </div>
      </PopoverContent>
    </Popover>
  )
}