import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: "IRVision: See heat in colour",
  description:
    "Colorization and semantic validation of thermal-infrared satellite imagery. U-Net colorizer trained on Landsat 8/9, evaluated on a city it never saw.",
};

export const viewport: Viewport = {
  themeColor: "#000000",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${inter.variable} antialiased`}>
      <body className="min-h-dvh bg-ink text-text">{children}</body>
    </html>
  );
}
