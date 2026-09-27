"use client";

import { AnimatePresence, motion } from "motion/react";
import { CircleCheck, LoaderCircle, Play, RotateCcw, TriangleAlert, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchExamples, runExample, runUpload, type Example, type Options } from "@/lib/api";
import { useSimulation } from "@/lib/useSimulation";
import { useMode } from "./ModeProvider";
import { PipelineFlow } from "./PipelineFlow";
import { Results } from "./Results";
import { Section, Toggle } from "./ui";

type Source = { kind: "example"; id: string } | { kind: "upload" };

export function Playground() {
  const { mode } = useMode();
  const [examples, setExamples] = useState<Example[]>([]);
  const [source, setSource] = useState<Source>({ kind: "example", id: "hyderabad_farmland" });
  const [options, setOptions] = useState<Options>({ superResolution: false, detection: false });
  const [irFile, setIrFile] = useState<File | null>(null);
  const [refFile, setRefFile] = useState<File | null>(null);
  const { run, start, reset } = useSimulation();
  const logRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchExamples(mode).then(setExamples).catch(() => setExamples([]));
  }, [mode]);

  useEffect(() => {
    reset(options);
  }, [options, reset]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [run.log.length]);

  useEffect(() => {
    if (run.phase === "done") setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 250);
  }, [run.phase]);

  const running = run.phase === "running";
  const canRun = !running && (source.kind === "example" || (mode === "live" && irFile !== null));

  function go() {
    if (!canRun) return;
    const request =
      source.kind === "example" ? runExample(mode, source.id, options) : runUpload(irFile as File, refFile, options);
    start(request, options);
  }

  const selectedTitle = useMemo(
    () => (source.kind === "example" ? examples.find((e) => e.id === source.id)?.title ?? "" : irFile?.name ?? "your image"),
    [source, examples, irFile],
  );

  return (
    <Section
      id="playground"
      eyebrow="Try it"
      title="Watch the pipeline work."
      lead={
        mode === "live"
          ? "Connected to the live model. Choose a scene from the held-out city, or upload your own thermal image."
          : "Choose a scene from Hyderabad, a city the model never saw. Results are real outputs of the trained models, pre-computed for this demo."
      }
    >
      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        {/* ---------------------------------------------------------------- controls */}
        <div className="card flex flex-col gap-5 p-5">
          <div>
            <p className="mb-3 text-[13px] font-medium uppercase tracking-wider text-muted">Input</p>
            <div className="grid grid-cols-3 gap-2.5">
              {examples.map((ex) => {
                const selected = source.kind === "example" && source.id === ex.id;
                return (
                  <button
                    key={ex.id}
                    type="button"
                    disabled={running}
                    onClick={() => setSource({ kind: "example", id: ex.id })}
                    className={`group relative overflow-hidden rounded-2xl border text-left transition ${
                      selected ? "border-neon neon-glow" : "border-line hover:border-line-strong"
                    }`}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={ex.ir_thumbnail} alt="" className="aspect-square w-full object-cover transition duration-500 group-hover:opacity-0" />
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={ex.thumbnail} alt="" className="absolute inset-0 aspect-square w-full object-cover opacity-0 transition duration-500 group-hover:opacity-100" />
                    <span className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 to-transparent px-2 pb-1.5 pt-5 text-[12px] font-medium">
                      {ex.title}
                    </span>
                  </button>
                );
              })}
            </div>
            {source.kind === "example" && (
              <p className="mt-3 text-[13px] leading-snug text-muted">
                {examples.find((e) => e.id === source.id)?.description}
              </p>
            )}
          </div>

          <div>
            <button
              type="button"
              disabled={mode !== "live" || running}
              onClick={() => setSource({ kind: "upload" })}
              className={`w-full rounded-2xl border border-dashed p-4 text-left transition ${
                source.kind === "upload" ? "border-neon bg-neon-soft" : "border-line-strong hover:border-neon/50"
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              <span className="flex items-center gap-2 text-[15px] font-medium">
                <Upload className="size-4 text-neon" /> Upload your own
              </span>
              <span className="mt-1 block text-[13px] text-muted">
                {mode === "live" ? "GeoTIFF, PNG or JPEG · single thermal band" : "Available when the live model server is connected"}
              </span>
            </button>
            {source.kind === "upload" && mode === "live" && (
              <div className="mt-3 space-y-2 text-[13px]">
                <FilePick label="Thermal image" file={irFile} onPick={setIrFile} accept=".tif,.tiff,.png,.jpg,.jpeg" />
                <FilePick label="True colour (optional, enables scores)" file={refFile} onPick={setRefFile} accept=".tif,.tiff,.png,.jpg,.jpeg" />
              </div>
            )}
          </div>

          <div className="space-y-2.5 border-t border-line pt-4">
            <Toggle
              label="Super-resolution ×2"
              hint="EDSR upscaling before colorization. Lowers accuracy on this data."
              checked={options.superResolution}
              disabled={running}
              onChange={(v) => setOptions((o) => ({ ...o, superResolution: v }))}
            />
            <Toggle
              label="Object detection"
              hint="YOLOv8 aerial model. Objects are tiny at 30 m per pixel."
              checked={options.detection}
              disabled={running}
              onChange={(v) => setOptions((o) => ({ ...o, detection: v }))}
            />
          </div>

          <button
            type="button"
            onClick={go}
            disabled={!canRun}
            className="mt-auto inline-flex items-center justify-center gap-2 rounded-full bg-neon px-6 py-3 text-[15px] font-semibold text-ink transition hover:bg-neon-strong hover:shadow-[0_0_32px_rgb(34_211_238/0.45)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {running ? <LoaderCircle className="size-4 animate-spin" /> : run.phase === "done" ? <RotateCcw className="size-4" /> : <Play className="size-4 fill-ink" />}
            {running ? "Processing…" : run.phase === "done" ? "Run again" : "Run pipeline"}
          </button>
        </div>

        {/* ---------------------------------------------------------------- live flow */}
        <div className="card relative flex min-h-[360px] flex-col overflow-hidden p-5 sm:p-6">
          {running && <div className="shimmer pointer-events-none absolute inset-x-0 top-0 h-px" />}
          <div className="mb-4 flex items-center justify-between gap-3">
            <div>
              <p className="text-[13px] font-medium uppercase tracking-wider text-muted">Pipeline</p>
              <p className="mt-1 text-lg font-semibold tracking-tight">
                {run.phase === "idle" && "Ready"}
                {running && <>Processing <span className="text-neon">{selectedTitle}</span></>}
                {run.phase === "done" && <>Finished in <span className="text-neon">{fmt(run.payload?.metrics.total_ms ?? 0)}</span></>}
                {run.phase === "error" && <span className="text-danger">Something went wrong</span>}
              </p>
            </div>
            {run.phase === "done" && <CircleCheck className="size-7 text-neon" />}
          </div>

          <PipelineFlow states={run.states} times={run.times} compact />

          <div
            ref={logRef}
            className="mt-4 max-h-56 flex-1 overflow-y-auto rounded-2xl border border-line bg-black/60 p-4 font-mono text-[12.5px] leading-6"
          >
            {run.phase === "idle" && <p className="text-muted">$ irvision process --input {selectedTitle || "…"}  # press Run</p>}
            <AnimatePresence initial={false}>
              {run.log.map((line) => (
                <motion.p
                  key={line.key}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex justify-between gap-4"
                >
                  <span className={line.note ? "text-muted" : "text-text"}>
                    <span className={line.note ? "text-muted" : "text-neon"}>{line.note ? "–" : "✓"}</span> {line.label}
                  </span>
                  <span className="text-muted">{line.note ?? (line.ms !== undefined ? fmt(line.ms) : "")}</span>
                </motion.p>
              ))}
            </AnimatePresence>
            {running && run.active && (
              <p className="text-neon">
                <span className="animate-pulse">▍</span> {labelFor(run.active)}…
              </p>
            )}
            {run.phase === "error" && (
              <p className="mt-2 flex items-start gap-2 text-danger">
                <TriangleAlert className="mt-1 size-4 shrink-0" /> {run.error}
              </p>
            )}
          </div>
        </div>
      </div>

      <div ref={resultsRef} className="scroll-mt-20">
        <AnimatePresence>
          {run.phase === "done" && run.payload && (
            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            >
              <Results payload={run.payload} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </Section>
  );
}

function FilePick({ label, file, onPick, accept }: { label: string; file: File | null; onPick: (f: File | null) => void; accept: string }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 rounded-xl border border-line bg-surface-2 px-3 py-2.5 hover:border-line-strong">
      <span className="truncate">{file ? file.name : label}</span>
      <span className="shrink-0 text-neon">{file ? "Change" : "Choose"}</span>
      <input type="file" accept={accept} className="hidden" onChange={(e) => onPick(e.target.files?.[0] ?? null)} />
    </label>
  );
}

const fmt = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);

function labelFor(key: string) {
  const labels: Record<string, string> = {
    validation: "Validating input",
    normalization: "Normalizing temperatures",
    enhancement: "Enhancing contrast (CLAHE)",
    super_resolution: "Upscaling with EDSR",
    colorization: "Colorizing with the U-Net",
    segmentation: "Mapping land cover",
    detection: "Detecting objects",
    metrics: "Scoring against true colour",
    semantic_validation: "Checking land-cover agreement",
    detection_validation: "Matching detections",
  };
  return labels[key] ?? key;
}
