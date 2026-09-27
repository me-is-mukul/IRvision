"use client";

import { useCallback, useRef, useState } from "react";
import type { Options, Payload } from "./api";
import { STAGE_DEFS, type NodeState } from "./stages";

export interface LogLine {
  key: string;
  label: string;
  ms?: number;
  note?: string;
}

export interface RunState {
  phase: "idle" | "running" | "done" | "error";
  states: Record<string, NodeState>;
  times: Record<string, number>;
  active?: string;
  log: LogLine[];
  payload?: Payload;
  error?: string;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function initialStates(options: Options): Record<string, NodeState> {
  return Object.fromEntries(
    STAGE_DEFS.map((s) => [s.key, s.optional && !options[s.optional] ? "off" : "idle"]),
  ) as Record<string, NodeState>;
}

export const IDLE: RunState = { phase: "idle", states: initialStates({ superResolution: false, detection: false }), times: {}, log: [] };

/**
 * Animates the pipeline while a request runs.
 * Phase A (waiting for the backend): early stages advance at a steady pace; the flow holds on
 * colorization until the result arrives.
 * Phase B (result received): remaining stages replay using their measured durations.
 */
export function useSimulation() {
  const [run, setRun] = useState<RunState>(IDLE);
  const runId = useRef(0);

  const reset = useCallback((options: Options) => {
    runId.current += 1;
    setRun({ ...IDLE, states: initialStates(options) });
  }, []);

  const start = useCallback(async (request: Promise<Payload>, options: Options) => {
    const id = ++runId.current;
    const alive = () => runId.current === id;
    let payload: Payload | null = null;
    let failure: string | null = null;
    request.then(
      (p) => (payload = p),
      (e: unknown) => (failure = e instanceof Error ? e.message : String(e)),
    );

    const states = initialStates(options);
    const times: Record<string, number> = {};
    const log: LogLine[] = [];
    const push = (patch: Partial<RunState>) =>
      alive() && setRun((prev) => ({ ...prev, states: { ...states }, times: { ...times }, log: [...log], ...patch }));
    push({ phase: "running", active: undefined, payload: undefined, error: undefined });

    const order = STAGE_DEFS.filter((s) => states[s.key] !== "off");
    let i = 0;

    // Phase A: optimistic progress while the backend works
    while (payload === null && failure === null && i < order.length) {
      const stage = order[i];
      states[stage.key] = "active";
      push({ active: stage.key });
      if (stage.key === "colorization") {
        while (payload === null && failure === null) {
          await sleep(80);
          if (!alive()) return;
        }
        break;
      }
      await sleep(420);
      if (!alive()) return;
      states[stage.key] = "done";
      log.push({ key: stage.key, label: stage.label });
      i += 1;
    }
    while (payload === null && failure === null) {
      await sleep(80);
      if (!alive()) return;
    }

    if (failure !== null) {
      for (const k of Object.keys(states)) if (states[k] === "active") states[k] = "idle";
      push({ phase: "error", active: undefined, error: failure });
      return;
    }

    const result = payload as unknown as Payload;
    const measured = Object.fromEntries(result.stages.map((s) => [s.key, s]));
    // attach measured times to stages already shown as done
    for (const line of log) {
      const s = measured[line.key];
      if (s?.status === "done") {
        times[line.key] = s.ms;
        line.ms = s.ms;
      }
    }

    // Phase B: replay the remaining stages with measured durations
    for (; i < order.length; i++) {
      const stage = order[i];
      const s = measured[stage.key];
      if (!s || s.status !== "done") {
        states[stage.key] = "off";
        log.push({ key: stage.key, label: stage.label, note: s?.status === "skipped" ? "skipped" : "off" });
        push({ active: undefined });
        await sleep(90);
        continue;
      }
      states[stage.key] = "active";
      push({ active: stage.key });
      await sleep(clamp(s.ms * 1.1, 260, 1300));
      if (!alive()) return;
      states[stage.key] = "done";
      times[stage.key] = s.ms;
      log.push({ key: stage.key, label: stage.label, ms: s.ms });
    }
    push({ phase: "done", active: undefined, payload: result });
  }, []);

  return { run, start, reset };
}
