import {
  Crosshair,
  Gauge,
  Map as MapIcon,
  Palette,
  ScanLine,
  ScanSearch,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  type LucideIcon,
} from "lucide-react";

export interface StageDef {
  key: string;
  label: string;
  short: string;
  description: string;
  icon: LucideIcon;
  optional?: "superResolution" | "detection";
}

/** Pipeline stages in execution order (mirrors STAGES in irvision/api/payload.py). */
export const STAGE_DEFS: StageDef[] = [
  { key: "validation", label: "Validation", short: "Validate", description: "Single band, size limits, no-data mask", icon: ShieldCheck },
  { key: "normalization", label: "Normalization", short: "Normalize", description: "Robust 2–98 % percentile stretch", icon: SlidersHorizontal },
  { key: "enhancement", label: "Enhancement", short: "Enhance", description: "CLAHE local contrast (16-bit)", icon: ScanLine },
  { key: "super_resolution", label: "Super-resolution", short: "Upscale", description: "EDSR ×2 upscaling", icon: Sparkles, optional: "superResolution" },
  { key: "colorization", label: "Colorization", short: "Colorize", description: "U-Net: thermal → RGB", icon: Palette },
  { key: "segmentation", label: "Land-cover map", short: "Segment", description: "DeepLabV3 on the colorized image", icon: MapIcon },
  { key: "detection", label: "Object detection", short: "Detect", description: "YOLOv8-OBB aerial classes", icon: Crosshair, optional: "detection" },
  { key: "metrics", label: "Quality metrics", short: "Score", description: "PSNR and SSIM vs. true colour", icon: Gauge },
  { key: "semantic_validation", label: "Semantic check", short: "Verify", description: "Land cover: colorized vs. true", icon: ScanSearch },
  { key: "detection_validation", label: "Detection check", short: "Match", description: "Objects: colorized vs. true", icon: Target, optional: "detection" },
];

/** Visual state of one stage in the flow. */
export type NodeState = "idle" | "active" | "done" | "off";
