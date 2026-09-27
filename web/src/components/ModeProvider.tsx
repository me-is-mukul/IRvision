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
  const [info, setInfo] = useState<ModeInfo>({ mode: "demo", checking: true, configured: "demo" });

  // the server URL can come from the page URL or browser storage, so resolve it on the client
  useEffect(() => {
    const configured = configuredMode();
    let cancelled = false;   // checkBackend() resolves to false at once when no server is configured
    checkBackend().then((ok) => !cancelled && setInfo({ mode: ok ? "live" : "demo", checking: false, configured }));
    return () => {
      cancelled = true;
    };
  }, []);

  return <ModeContext.Provider value={info}>{children}</ModeContext.Provider>;
}

export const useMode = () => useContext(ModeContext);
