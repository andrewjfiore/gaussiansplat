"use client";
import Link from "next/link";
import { ThemeProvider, useTheme } from "./ThemeProvider";
import { ToastProvider } from "./Toast";
import { Sun, Moon } from "lucide-react";

function Header() {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="border-b border-gray-800 bg-gray-950">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <Link
          href="/"
          className="text-xl font-bold text-white hover:text-blue-400 transition"
        >
          GaussianSplat Studio
        </Link>
        <nav className="flex items-center gap-4 text-sm">
          <Link
            href="/"
            className="text-gray-400 hover:text-white transition"
          >
            Projects
          </Link>
          <Link
            href="/settings"
            className="text-gray-400 hover:text-white transition"
          >
            Settings
          </Link>
          <button
            onClick={toggleTheme}
            className="text-gray-400 hover:text-white transition p-1 rounded-md hover:bg-gray-800"
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? (
              <Sun className="w-4 h-4" />
            ) : (
              <Moon className="w-4 h-4" />
            )}
          </button>
        </nav>
      </div>
    </header>
  );
}

export function ClientProviders({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <ToastProvider>
        <Header />
        <main className="max-w-7xl mx-auto px-4 py-6">{children}</main>
      </ToastProvider>
    </ThemeProvider>
  );
}
