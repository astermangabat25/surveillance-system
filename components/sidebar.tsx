"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { 
  LayoutDashboard, 
  Video, 
  User
} from "lucide-react"

const navItems = [
  { icon: Video, label: "Surveillance", href: "/" },
  { icon: LayoutDashboard, label: "Dashboard", href: "/dashboard" },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="flex flex-col w-16 bg-sidebar border-r border-sidebar-border h-full">
      {/* Logo */}
      <div className="flex items-center justify-center h-16 border-b border-sidebar-border">
        <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
          <span className="text-primary-foreground font-bold text-sm">AL</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 flex flex-col items-center py-4 gap-2">
        {navItems.map((item) => {
          const isActive = pathname === item.href || 
            (item.href === "/" && pathname.startsWith("/video")) ||
            (item.href === "/dashboard" && pathname === "/search")
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center justify-center w-10 h-10 rounded-lg transition-all duration-200",
                isActive
                  ? "bg-primary/20 text-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary"
              )}
              title={item.label}
            >
              <item.icon className="w-5 h-5" />
            </Link>
          )
        })}
      </nav>

      {/* Profile Avatar */}
      <div className="flex items-center justify-center pb-4">
        <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center border border-border">
          <User className="w-5 h-5 text-muted-foreground" />
        </div>
      </div>
    </aside>
  )
}
