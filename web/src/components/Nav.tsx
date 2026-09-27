"use client";

import { Satellite } from "lucide-react";
import { useMode } from "./ModeProvider";

const LINKS = [
  { href: "#flow", label: "How it works" },
  { href: "#playground", label: "Try it" },
  { href: "#results", label: "Results" },
  { href: "#impact", label: "Who it helps" },
];

export function Nav() {
  const { mode, checking } = useMode();
  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-line bg-ink/60 backdrop-blur-xl backdrop-saturate-150">
      <nav className="mx-auto flex h-14 max-w-6xl items-center justify-between px-5">
        <a href="#top" className="flex items-center gap-2 text-[15px] font-semibold tracking-tight">
          <span className="grid size-7 place-items-center rounded-lg bg-neon text-ink">
            <Satellite className="size-4" strokeWidth={2.25} />
          </span>
          IRVision
        </a>
        <div className="hidden items-center gap-5 text-[13px] text-subtle md:flex">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href} className="transition-colors hover:text-text">
              {l.label}
            </a>
          ))}
        </div>
        <span
          className={`flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${
            mode === "live" ? "border-neon/40 text-neon" : "border-line-strong text-subtle"
          }`}
          title={mode === "live" ? "Connected to the IRVision model server" : "Showing pre-computed real results"}
        >
          <span className={`size-1.5 rounded-full ${mode === "live" ? "bg-neon shadow-[0_0_8px_rgb(34_211_238)]" : "bg-muted"}`} />
          {checking ? "Connecting…" : mode === "live" ? "Live model" : "Demo mode"}
        </span>
      </nav>
    </header>
  );
}
