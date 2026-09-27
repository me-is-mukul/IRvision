export function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto grid max-w-6xl gap-6 px-5 py-8 text-[13px] text-muted sm:grid-cols-[1fr_auto]">
        <div className="max-w-xl space-y-2 leading-relaxed">
          <p className="text-[15px] font-semibold text-text">IRVision</p>
          <p>
            The colorized output is a learned, plausible reconstruction of true colour, not a physical measurement.
          </p>
          <p>
            Landsat 8/9 imagery courtesy of the U.S. Geological Survey via the Microsoft Planetary Computer. ESA
            WorldCover 2021 © ESA WorldCover project, contains modified Copernicus Sentinel data. Pre-trained models:
            EDSR (super-image), YOLOv8 (Ultralytics), VGG19 and DeepLabV3 (torchvision).
          </p>
        </div>
        <div className="flex gap-4 sm:flex-col sm:gap-2 sm:text-right">
          <a href="#flow" className="hover:text-text">How it works</a>
          <a href="#playground" className="hover:text-text">Try it</a>
          <a href="#results" className="hover:text-text">Results</a>
        </div>
      </div>
    </footer>
  );
}
