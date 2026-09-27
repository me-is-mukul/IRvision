"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { checkBackend, configuredMode, type Mode } from "@/lib/api";

interface ModeInfo {
  /** "live" only when a backend is configured AND reachable */
  mode: Mode;
  checking: boolean;
  configured: Mode;
}

const ModeContext = createContext<ModeInfo>({ mode: "demo", checking: false, configured: "demo" });

export function ModeProvider({ children }: { children: ReactNode }) {
  const configured = configuredMode();
  const [info, setInfo] = useState<ModeInfo>({ mode: "demo", checking: configured === "live", configured });

  useEffect(() => {
    if (configured !== "live") return;
    let cancelled = false;
    checkBackend().then((ok) => !cancelled && setInfo({ mode: ok ? "live" : "demo", checking: false, configured }));
    return () => {
      cancelled = true;
    };
  }, [configured]);

  return <ModeContext.Provider value={info}>{children}</ModeContext.Provider>;
}

export const useMode = () => useContext(ModeContext);
