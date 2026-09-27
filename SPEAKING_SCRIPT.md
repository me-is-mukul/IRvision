# IRVision: Presentation Script

**Length:** about 5 minutes, plus Q&A.

**Before you start:** open the dashboard and press **Process** once, so all models are loaded.

```powershell
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

---

## 1. Opening: the problem (≈ 45 s)

> "Satellites don't only take normal photos. Many carry **thermal-infrared sensors**, which measure heat instead of light. Thermal sensing has real advantages: it measures surface temperature directly, and thermal sensors can image at night, when ordinary cameras see nothing.
>
> The problem is that a thermal image is a single grey band of temperatures. For most people, and even for many analysts, it's hard to read. Is that dark patch a lake, a forest or a shadow? You need training to interpret it.
>
> So we asked: **can we turn a thermal satellite image into a normal-looking colour image that anyone can understand, and can we prove how trustworthy that colour image is?**"

---

## 2. What we built (≈ 60 s)

> "That's IRVision. You give it a thermal satellite image, and it produces an enhanced version and a colour version, and then checks its own work.
>
> The pipeline has four parts:
>
> 1. **Enhancement.** We clean the image, mask clouds and missing data, and boost local contrast so hidden detail becomes visible.
> 2. **Colorization.** A deep-learning model, a U-Net, predicts the true-colour image from the thermal band. We trained it on real Landsat 8 and 9 satellite data: the thermal band as input, and the real red-green-blue image of the same moment as the answer.
> 3. **Semantic validation.** This is what makes our project different. A pretty picture isn't enough; it has to *mean* the right thing. So a second model classifies land cover (water, trees, crops, buildings, bare ground) in both our colour image and the real one, and we measure how well they agree.
> 4. **Optional extras.** Super-resolution and object detection, which we tested honestly; more on that in a moment."

---

## 3. Live demo (≈ 90 s)

**Action:** sidebar → **Example Landsat scene** → **hyderabad (held-out)** → crop **1024** over lakes and farmland → **▶ Process**.

> "This is Hyderabad. The model **never saw this city** during training. We kept it aside from the start as a genuine test.
>
> From left to right: the raw thermal image, the enhanced version, our colorized output, and the real colour photo taken by the satellite at the same moment.
>
> Up here are the scores: PSNR measures colour accuracy, SSIM measures structural similarity, plus the processing time. The whole 60 × 60 km scene takes **about 1.6 seconds** on a laptop GPU.
>
> Below is the land-cover check: what our model's image shows next to what reality shows. Water and vegetation line up well."

**Optional action:** switch to **Upload an IR image** → upload `demo/hyderabad_city_ir.png` as the IR image and `demo/hyderabad_city_truecolor.png` as the reference → **▶ Process**.

> "And this is where we're honest. In the dense city centre, agreement drops to about 7%: our model paints buildings as if they were vegetation. The colour scores alone would never have told us that. **Our validation layer caught it.**"

---

## 4. Results (≈ 45 s)

**Action:** open the **Model report** tab.

> "We compared against simple baselines on 355 test patches:
>
> - Our model has the **best structural similarity on every test set**, including the unseen city.
> - For colour accuracy it clearly wins in known cities, and on the unseen city it matches the best simple method.
> - The real difference is in meaning: our images preserve about **eight times more land-cover information** than the best simple baseline.
>
> We also tested the advanced features properly. Super-resolution and object detection **made results worse** at Landsat's 30-metre resolution, so they're available as options but switched off by default. We'd rather show a measured 'no' than an unmeasured 'yes'."

---

## 5. Who this helps (≈ 45 s)

> "Who can use this?
>
> - **Disaster-response teams.** Thermal imaging keeps working at night, when a visible image would be dark. A colorized view makes a thermal image of a flood-affected region quick to read for responders who aren't remote-sensing experts. Water is exactly the class our model preserves best.
> - **Agriculture and water management.** Crops, forests and water bodies are preserved well, and thermal data already signals irrigation and crop stress. Colorization makes those images easier to share with farmers and local officials.
> - **Environmental and climate researchers**, and anyone working on urban heat: a readable view of thermal imagery alongside the temperature data.
> - **Education and public communication.** Satellite data becomes understandable to students and the general public.
>
> And just as important is who it should *not* be used for yet: mapping city extent, because our own validation shows buildings aren't preserved. We'd rather tell users that than let them find out the hard way."

---

## 6. Closing (≈ 20 s)

> "To summarise: IRVision turns hard-to-read thermal satellite images into colour images, **and measures whether they can be trusted**. It's tested on a city it never saw, and it's honest about where it works and where it doesn't. Next, we want to fix urban areas with a land-cover-aware training loss and add more seasons and regions. Thank you."

---

## Q&A preparation

### About the project

**Q: What exactly is the input and output?**
> Input: a single-band thermal image in any unit (Kelvin, raw sensor values, PNG or GeoTIFF). Output: an enhanced thermal image, a colour image, a land-cover map and quality scores.

**Q: Where does your data come from?**
> Landsat 8 and 9 Level-2 data from the Microsoft Planetary Computer. It's free public data: ten scenes, nine for training and one city, Hyderabad, kept completely separate for testing. Land-cover labels come from ESA WorldCover 2021.

**Q: Is the colour image real?**
> No, and we say so clearly. It's a learned, plausible reconstruction, not a measurement. That's exactly why we built the validation step.

**Q: Why not just use a colour map, like thermal cameras do?**
> We tested that. A pseudo-colour map scores around 10 dB because it isn't trying to show real colours. It's useful for experts reading temperature, but it doesn't look like the real world. Our output does, and we measured how closely.

### About the results

**Q: How do you know the colours are right?**
> We compare against the real colour image captured at the same moment, using PSNR and SSIM, and we check land-cover agreement. All reported numbers come from evaluation scripts, not from hand-picked examples.

**Q: How does it do on new places?**
> On the unseen city, structure and land-cover agreement are clearly better than the baselines. Colour accuracy matches the best simple method there, but doesn't beat it yet. More diverse training data is the fix. Going from three to nine cities was already our biggest improvement.

**Q: What's your biggest weakness?**
> Dense urban areas: buildings come out looking like vegetation. Our semantic check found this, and fixing it is our top next step.

**Q: Why did you drop super-resolution and object detection?**
> We didn't drop them; we implemented and measured them. Super-resolution lowered every score, because the model was trained on 30 m pixels. Object detection matched zero real objects on colorized images, since most objects are only a few pixels wide at 30 m. Both are optional switches, off by default, and would make sense with higher-resolution sensors.

### About scope and use

**Q: Could this work at night?**
> Thermal sensors do capture night images, and that's a key motivation. Our current model was trained on daytime scenes, where a true-colour answer exists for comparison. Night-time imagery is a natural next step, but we haven't validated it yet.

**Q: Why only Indian cities?**
> Mostly Indian cities, plus the Nile Delta, all in the dry season. It was a deliberate, manageable scope for a hackathon. Adding a region is one line in the configuration file.

### About the engineering

**Q: How fast is it?**
> About 11 milliseconds per 256 × 256 tile on the GPU, and about 1.6 seconds for a full 2048 × 2048 scene.

**Q: What hardware do you need?**
> We trained on a laptop RTX 4050 with 6 GB. Inference also runs on CPU, just slower.

**Q: How reliable is the software?**
> 81 automated tests cover data loading, alignment, models, metrics, the pipeline and the dashboard. If a model file is missing, the app falls back to a simpler method and shows a warning instead of crashing.

**Q: What would you do with more time?**
> Fix urban areas with a land-cover-aware loss, add more seasons and continents, try the Lab colour space, and test on higher-resolution thermal sensors.

---

## Checklist before presenting

- [ ] Dashboard started with the project's Python (`.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py`)
- [ ] **Process** pressed once beforehand, so the models are loaded
- [ ] `demo/` files ready for the upload step
- [ ] Backup figures open, in case of problems: `outputs/results/model_comparison.png`, `outputs/results/semantic_comparison.png`
- [ ] Timing rehearsed. For a 3-minute slot, shorten section 2 and skip the optional upload in section 3.
- [ ] Hackathon rules on AI-assisted development checked. If disclosure is required, add one sentence to the closing.

---

## GitHub repository details

**Description:**

```
Thermal-infrared to RGB colorization for Landsat 8/9 satellite imagery. U-Net colorizer with CLAHE enhancement, land-cover semantic validation, optional super-resolution and object detection, and an interactive Streamlit dashboard, evaluated on a held-out city.
```

**Topics:** `remote-sensing` `satellite-imagery` `thermal-infrared` `landsat` `image-colorization`
`deep-learning` `pytorch` `unet` `semantic-segmentation` `super-resolution` `object-detection`
`yolov8` `computer-vision` `earth-observation` `geospatial` `streamlit` `rasterio` `python`
