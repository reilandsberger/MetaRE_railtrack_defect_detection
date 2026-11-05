// rail3D update slides, 2026-11-05 -- styled to sit after rail3D_overview.pptx (13.33 x 7.5, Calibri,
// grey kicker over a 29 pt title, navy ink, blue accent).
//
//   npm install pptxgenjs            (deck tooling only, like python-pptx for update_deck.py)
//   node presentation/slides_2026-11-05.js
//
// Writes data/generated/2026-11-05/rail3D_update_2026-11-05.pptx: 13 slides (setup, previous work,
// data generation, training, pruning, prelim result, where it fails, detection rules, the setup
// sweep). Dashed "FIGURE TO ADD" boxes name the file and folder of each figure tonight's runs make.
// Numbers are from the 2026-11-05 lab returns; speaker notes give each source.
const pptxgen = require("pptxgenjs");
const path = require("path");

const RAIL = "C:/Users/Rei/Documents/MetaRE_railtrack_defect_detection/rail3D";
const FIG = RAIL + "/data/figures";
const SWEEP = RAIL + "/data/generated/2026-11-05/sweep_F1_f5be4bbb";   // sweep_06a = panel A of sweep_06
const OUT = RAIL + "/data/generated/2026-11-05/rail3D_update_2026-11-05.pptx";

const INK = "1B2733", MUTE = "7B8494", ACC = "1F6FEB", GOOD = "1A7F5A", BAD = "B42318";
const PANEL = "F3F5F8", LINE = "C9CFD8", WHITE = "FFFFFF", AMBER = "B26B00", PALE = "DCE8FD";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "rail3D update 2026-11-05";
pres.author = "Rei Landsberger";
pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };

const W = 13.333, M = 0.6, CW = W - 2 * M;

function header(s, kicker, title) {
  s.background = { color: WHITE };
  s.addText(kicker, { x: M, y: 0.32, w: CW, h: 0.32, fontSize: 11.5, bold: true, color: MUTE,
    charSpacing: 1, margin: 0, isTextBox: true, objectName: "kicker" });
  s.addText(title, { x: M, y: 0.58, w: CW, h: 0.62, fontSize: 29, bold: true, color: INK,
    margin: 0, isTextBox: true, objectName: "title" });
  s.addShape(pres.shapes.RECTANGLE, { x: M + 0.02, y: 1.22, w: 1.05, h: 0.05, fill: { color: ACC },
    line: { color: ACC, width: 0 }, objectName: "title mark" });
}

function placeholder(s, x, y, w, h, file, from, what) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: "F7F9FC" },
    line: { color: "9AA3B2", width: 1.25, dashType: "dash" }, objectName: "figure placeholder: " + file });
  s.addText([
    { text: "FIGURE TO ADD", options: { fontSize: 12, bold: true, color: ACC, breakLine: true } },
    { text: file, options: { fontSize: 16, bold: true, color: INK, breakLine: true } },
    { text: from, options: { fontSize: 11, color: MUTE, breakLine: true } },
    { text: what, options: { fontSize: 12, color: INK, italic: true } },
  ], { x: x + 0.3, y: y + 0.2, w: w - 0.6, h: h - 0.4, valign: "middle", align: "center",
       paraSpaceAfter: 4, isTextBox: true, objectName: "placeholder text" });
}

function card(s, x, y, w, h, head, body, color = INK, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: PANEL },
    line: { color: PANEL, width: 0 }, objectName: "card " + head });
  const runs = [{ text: head, options: { fontSize: opts.headSize || 16, bold: true, color,
    breakLine: true } }];
  (Array.isArray(body) ? body : [body]).forEach((t, i, a) => runs.push({ text: t,
    options: { fontSize: opts.bodySize || 14, color: INK, breakLine: i < a.length - 1,
      bullet: opts.bullets ? { indent: 14 } : false } }));
  s.addText(runs, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: h - 0.24, valign: "top",
    paraSpaceAfter: opts.space ?? 5, isTextBox: true, objectName: "card text " + head });
}

function stat(s, x, y, w, big, label, color = ACC, bigSize = 40) {
  s.addText(big, { x, y, w, h: 0.75, fontSize: bigSize, bold: true, color, margin: 0,
    valign: "bottom", isTextBox: true, objectName: "stat " + big });
  s.addText(label, { x, y: y + 0.8, w, h: 0.75, fontSize: 13, color: INK, margin: 0, valign: "top",
    isTextBox: true, objectName: "stat label " + big });
}

function dot(s, x, y, d, n, color = ACC) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color, width: 0 } });
  s.addText(String(n), { x, y, w: d, h: d, fontSize: 14, bold: true, color: WHITE, align: "center",
    valign: "middle", margin: 0, isTextBox: true });
}

const AX = { catAxisLabelColor: INK, valAxisLabelColor: INK, catAxisLabelFontSize: 12,
  valAxisLabelFontSize: 11, catAxisLabelFontFace: "Calibri", valAxisLabelFontFace: "Calibri",
  valGridLine: { color: "E3E7ED", size: 0.75 }, catGridLine: { style: "none" },
  titleFontFace: "Calibri", titleFontSize: 14, titleColor: INK, showTitle: true,
  dataLabelFontFace: "Calibri", dataLabelFontSize: 11, dataLabelColor: INK,
  legendFontFace: "Calibri", legendFontSize: 12, legendColor: INK };

// ---------------------------------------------------------------- 1. divider
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addText("UPDATE  ·  5 NOVEMBER 2026", { x: M, y: 1.7, w: CW, h: 0.4, fontSize: 14, bold: true,
    color: "8FB4F8", charSpacing: 2, margin: 0, isTextBox: true });
  s.addText("Choosing the setup, and training a detector-only system", { x: M, y: 2.15, w: 10.5, h: 1.5,
    fontSize: 40, bold: true, color: WHITE, margin: 0, valign: "top", isTextBox: true });
  const parts = ["Setup and previous work", "Data generation", "Training and pruning",
    "Detection results", "The setup sweep"];
  parts.forEach((t, i) => {
    const x = M + i * 2.45;
    dot(s, x, 4.55, 0.5, i + 1, ACC);
    s.addText(t, { x, y: 5.15, w: 2.2, h: 0.7, fontSize: 15, color: "E6EBF2", margin: 0, valign: "top",
      isTextBox: true });
  });
  s.addNotes("Section opener for the 2026-11-05 update. The deck before this point describes the " +
    "simulator; this section covers what changed: the detector-only focus, data generation, " +
    "training with pruning, and the setup sweep.");
}

// ---------------------------------------------------------------- 2. setup
{
  const s = pres.addSlide();
  header(s, "SETUP", "One horn, one observation plane, eight detectors");
  s.addImage({ path: FIG + "/setup_diagram.png", x: M, y: 1.45, w: CW, h: CW / (3400 / 1100),
    altText: "rail3D simulated setup: 3D scene, side view and top view" });
  const y = 5.55, cw = (CW - 3 * 0.3) / 4;
  [["60 GHz", "λ = 5 mm; every distance below is set in wavelengths"],
   ["55°", "RFspin H-A75-W20 horn, 278.5 mm out, lowest edge level with the plane"],
   ["150 mm", "plane above the crown: 150 × 75 mm, 60 × 30 points"],
   ["130 → 8", "detectors 100 mm above the plane, pruned during training"]]
    .forEach(([b, l], i) => stat(s, M + i * (cw + 0.3), y - 0.15, cw, b, l, ACC, 34));
  s.addNotes("Figure: data/figures/setup_diagram.png (2026-10-13, after the horn move). " +
    "The horn hits at 55°, so the crown's mirror reflection lands at x = -214 mm, outside today's " +
    "150 mm plane: the system is dark-field and sees defects as scattered light (deck slide 13).");
}

// ---------------------------------------------------------------- 3. previous work
{
  const s = pres.addSlide();
  header(s, "PREVIOUS WORK", "The simulator is verified; the open question is where to look");
  const items = [
    ["24 Sep", "Real horn modelled: RFspin H-A75-W20, 20.09 dBi, matching its datasheet"],
    ["2 Oct", "Horn moved out to 278.5 mm, its lowest edge level with the plane"],
    ["13 Oct", "Wear rebuilt as a smooth wheel cut on the gauge corner"],
    ["29 Oct", "Design study: today's plane sees only ~10 mm of the crown-to-gauge arc"],
    ["2 Nov", "Full-wave targets: cracks act as waveguides; wear, dents, shells model well"],
    ["5 Nov", "All 13 verification gates pass; new 8,400-rail dataset trained"]];
  const x0 = M, y0 = 1.6, dy = 0.86;
  s.addShape(pres.shapes.LINE, { x: x0 + 0.17, y: y0 + 0.2, w: 0, h: dy * (items.length - 1),
    line: { color: LINE, width: 2 } });
  items.forEach(([d, t], i) => {
    const y = y0 + i * dy, last = i === items.length - 1;
    s.addShape(pres.shapes.OVAL, { x: x0 + 0.05, y: y + 0.08, w: 0.24, h: 0.24,
      fill: { color: last ? GOOD : ACC }, line: { color: WHITE, width: 2 } });
    s.addText(d, { x: x0 + 0.5, y, w: 1.0, h: 0.4, fontSize: 14, bold: true, color: last ? GOOD : ACC,
      margin: 0, valign: "middle", isTextBox: true });
    s.addText(t, { x: x0 + 1.5, y, w: 5.6, h: 0.62, fontSize: 14, color: INK, margin: 0, valign: "top",
      isTextBox: true });
  });
  const sx = 8.0, sw = W - M - sx;
  stat(s, sx, 1.45, sw, "13 / 13", "verification gates pass with the current horn and wear model", GOOD);
  stat(s, sx, 3.2, sw, "46–100%", "rail3D inside the full-wave agreement gates: wear and shell 100%, " +
    "dents 89–94%, cracks 46–67%", ACC);
  stat(s, sx, 4.95, sw, "0.67 → 0.92", "field-level recall on the crown-to-gauge arc: today's plane vs " +
    "its far edge extended to x = −190 mm", ACC);
  s.addNotes("Sources: README findings 29, 31, 33, 34; DESIGN_STUDY.md; lab_report 2026-11-05 " +
    "(V3 0.0010273 unchanged, V6 1.2% < 2%, V7 min cosine 0.982, V9 20.09 dBi). The full-wave numbers " +
    "are 2D exact solver vs rail3D on 162 cross-sections (RAIL_TARGETS.md).");
}

// ---------------------------------------------------------------- 4. data generation
{
  const s = pres.addSlide();
  header(s, "DATA GENERATION", "8,400 simulated rails in 35 minutes on one GPU");
  const steps = [
    ["Draw a defect", "crack, dent, wear or shell, sized over your operating ranges"],
    ["Mesh the railhead", "λ/8 = 0.625 mm facets; a random pose (2° roll, 4 mm shift)"],
    ["Cast shadows", "a coarse λ/2 occluder hides facets the horn cannot see"],
    ["Solve the reflected field", "physical optics on the plane, upward light only"],
    ["Store", "the complex field + every defect parameter, for training and analysis"]];
  steps.forEach(([h, t], i) => {
    const y = 1.5 + i * 0.78;
    dot(s, M, y + 0.05, 0.42, i + 1, ACC);
    s.addText([{ text: h, options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
      { text: t, options: { fontSize: 13, color: INK } }],
      { x: M + 0.6, y, w: 4.75, h: 0.74, margin: 0, valign: "top", isTextBox: true });
  });
  stat(s, M, 5.55, 2.4, "2,000", "rails per defect class, plus 400 intact", ACC, 32);
  stat(s, M + 2.75, 5.55, 2.4, "4.0 / s", "rails per second on the RTX 5090", ACC, 32);
  const iw = 6.75, ih = iw / (2340 / 1770), ix = W - M - iw;
  s.addImage({ path: FIG + "/defect_sections.png", x: ix, y: 1.4, w: iw, h: ih,
    altText: "Cross-sections of above-average crack, dent, wear and shell samples" });
  s.addText("Above-average samples of each class, cut where deepest (red = removed material)",
    { x: ix, y: 1.4 + ih + 0.02, w: iw, h: 0.35, fontSize: 11, italic: true, color: MUTE, margin: 0,
      isTextBox: true });
  s.addNotes("Dataset L5_prelim_689da5, commit 9a3a7d1, 0.58 h at 4.0 samples/s. Ranges: crack depth " +
    "4-8 mm, width 1.5-3 mm, length 10-50 mm, 3 lines; dent depth 1.5-8 mm, footprint 10-30 mm; wear " +
    "gauge cut 2-8 mm at 30-60°, crown wear in 50%; shell depth 1-5 mm, radius 4-10 mm. Figure: " +
    "data/figures/defect_sections.png (2026-10-13).");
}

// ---------------------------------------------------------------- 5. ML pipeline
{
  const s = pres.addSlide();
  header(s, "MACHINE LEARNING", "From reflected field to barcode to alarm");
  const boxes = [["Field on the plane", "60 × 30 complex values"],
    ["Propagate", "100 mm up, to the detector plane"],
    ["130 soft detectors", "13 × 10 windows, 92% of the plane"],
    ["Prune to 8", "keep what adds new information"],
    ["Barcode", "8 detector powers, normalised"],
    ["Alarm", "distance from intact above the 5% line"]];
  const bw = 1.78, gap = (CW - 6 * bw) / 5, by = 1.6, bh = 1.55;
  boxes.forEach(([h, t], i) => {
    const x = M + i * (bw + gap), last = i === boxes.length - 1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: by, w: bw, h: bh, rectRadius: 0.08,
      fill: { color: last ? GOOD : (i === 3 ? ACC : PALE) }, line: { color: WHITE, width: 0 } });
    const c = (last || i === 3) ? WHITE : INK;
    s.addText([{ text: h, options: { bold: true, fontSize: 15, color: c, breakLine: true } },
      { text: t, options: { fontSize: 12, color: c } }],
      { x: x + 0.1, y: by + 0.08, w: bw - 0.2, h: bh - 0.16, align: "center", valign: "middle",
        margin: 0, isTextBox: true });
    if (i < boxes.length - 1)
      s.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: x + bw + gap / 2 - 0.11, y: by + bh / 2 - 0.11,
        w: 0.22, h: 0.22, fill: { color: MUTE }, line: { color: MUTE, width: 0 }, rotate: 90 });
  });
  const cy = 3.55, ch = 3.0, cw = (CW - 2 * 0.3) / 3;
  card(s, M, cy, cw, ch, "What is trained", [
    "Where each soft detector window sits (it can move)",
    "Which 8 of the 130 survive",
    "No metasurface: on hold since the ablation"], ACC, { bullets: true, bodySize: 15 });
  card(s, M + cw + 0.3, cy, cw, ch, "How it is trained", [
    "Rank loss: every defect should score above every intact rail (a soft AUC)",
    "Extra weight on the hardest 10% of pairs; intact rails pulled together",
    "Detection only: no class loss",
    "1,500 epochs, about 2.5 min on the 5090"], ACC, { bullets: true, bodySize: 15 });
  card(s, M + 2 * (cw + 0.3), cy, cw, ch, "How the model is picked", [
    "Highest validation AUC, among epochs that already have 8 detectors",
    "Alarm line at 5% false alarms on the validation intact rails",
    "Scored once on a held-out test split"], ACC, { bullets: true, bodySize: 15 });
  s.addNotes("TrainConfig(surface='none', task='detect'), train_detect.py. The rank objective is a " +
    "pairwise soft-AUC hinge on normalised barcodes (README finding 10). Window sharpness tau is " +
    "annealed from w/4 to w/8 over 250 epochs, never below the pixel pitch. Selection only at the " +
    "final detector count is README finding 22: otherwise the 130-detector model is picked.");
}

// ---------------------------------------------------------------- 6. pruning
{
  const s = pres.addSlide();
  header(s, "PRUNING", "130 → 8 detectors, keeping the ones that add new information");
  const sched = [[0, 130], [55, 130], [60, 98], [65, 74], [70, 56], [75, 42], [80, 32], [85, 24],
    [90, 18], [95, 14], [100, 10], [105, 8], [150, 8]];
  s.addChart(pres.charts.LINE, [{ name: "detectors", labels: sched.map(p => String(p[0])),
    values: sched.map(p => p[1]) }], { x: M, y: 1.45, w: 6.2, h: 3.6, ...AX,
    title: "Detectors left vs training epoch (one prune every 5 epochs, keep 75%)",
    chartColors: [ACC], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    showLegend: false, showValue: true, dataLabelPosition: "t", dataLabelFontSize: 10,
    valAxisMinVal: 0, valAxisMaxVal: 140, catAxisTitle: "epoch", showCatAxisTitle: true,
    catAxisTitleFontSize: 12, catAxisTitleColor: INK });
  card(s, M, 5.3, 6.2, 1.6, "The rule", ["Each detector is worth  std × (1 − max |correlation| " +
    "with the detectors kept so far).  The least valuable go first."], ACC, { space: 4 });
  card(s, 7.1, 1.45, W - M - 7.1, 2.35, "Why not keep the most variable?", [
    "Overlapping windows see nearly the same light, so a duplicate looks as variable as the original.",
    "Test case (V0b): variance keeps [0, 4, 5], a duplicate; redundancy keeps [3, 4, 5]."],
    BAD, { space: 6 });
  placeholder(s, 7.1, 4.0, W - M - 7.1, 2.9, "Final 8-detector layout",
    "trained detector-only model, n_det = 8 (to plot from its checkpoint)",
    "Where the 8 surviving windows sit on the plane");
  s.addNotes("prune_criterion='redundancy' (README finding 17, V0b gate). Fraction schedule " +
    "n -> n - round(0.25 n) every 5 epochs from epoch 60, so 130 reaches 8 at epoch 105, inside the " +
    "60-150 window. train_detect.py --n-det 4 8 12 16 tests how many detectors detection needs.");
}

// ---------------------------------------------------------------- 7. prelim result
{
  const s = pres.addSlide();
  header(s, "PRELIM RESULT", "Detectors alone beat every trained metasurface");
  const runs = [["No metasurface", 0.950], ["Metasurface, flat start", 0.889],
    ["Flat start, 4× epochs", 0.885], ["No light-capture reward", 0.872],
    ["As shipped", 0.846], ["Flat, no capture reward", 0.843]];
  s.addChart(pres.charts.BAR, [{ name: "validation AUC", labels: runs.map(r => r[0]),
    values: runs.map(r => r[1]) }], { x: M, y: 1.45, w: 7.4, h: 5.4, barDir: "bar", ...AX,
    title: "Detection AUC, validation split (same data, schedule and seed)",
    chartColors: [GOOD, MUTE, MUTE, MUTE, MUTE, MUTE],
    showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    valAxisMinVal: 0.8, valAxisMaxVal: 1.0, valAxisLabelFormatCode: "0.00", catAxisOrientation: "maxMin",
    barGapWidthPct: 60 });
  const sx = 8.4, sw = W - M - sx;
  stat(s, sx, 1.45, sw, "0.950", "detection AUC with no metasurface, 8 detectors", GOOD, 44);
  card(s, sx, 3.15, sw, 1.75, "Optimisation, not physics", ["A flat metasurface is exactly the " +
    "no-metasurface system, so a trained one should never score lower."], INK, { space: 4 });
  card(s, sx, 5.1, sw, 1.8, "Decision", ["Metasurface on hold. Optimise the detector-only " +
    "system, for detection only."], GOOD, { space: 4 });
  s.addNotes("surface_ablation.json, L5_prelim_689da5. Flat start = slm_init_std 0; capture reward = " +
    "w_capture 0.2. README finding 24. The no-metasurface model was trained with the classification " +
    "loss on; train_detect.py retrains it for detection only.");
}

// ---------------------------------------------------------------- 8. where it fails
{
  const s = pres.addSlide();
  header(s, "WHERE IT FAILS", "Position on the railhead decides what is detected");
  const s0 = ["106", "115", "126", "137", "148", "158"];
  s.addChart(pres.charts.LINE, [
    { name: "crack", labels: s0, values: [0.0, 0.06, 0.06, 0.45, 0.91, 0.76] },
    { name: "dent", labels: s0, values: [0.09, 0.12, 0.27, 0.94, 1.0, 0.97] }],
    { x: M, y: 1.45, w: 6.0, h: 4.1, ...AX, title: "Detected at 5% false alarms vs arc position s0 (mm)",
      chartColors: [BAD, ACC], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
      showLegend: true, legendPos: "b", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%" });
  const ang = ["32", "38", "45", "50", "54", "58"];
  s.addChart(pres.charts.LINE, [{ name: "wear", labels: ang, values: [1.0, 0.97, 0.79, 0.76, 0.67, 0.44] }],
    { x: 6.9, y: 1.45, w: W - M - 6.9, h: 4.1, ...AX,
      title: "Wear detected vs gauge-cut angle (degrees from vertical)",
      chartColors: [AMBER], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
      showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.8, w: CW, h: 1.1, rectRadius: 0.08,
    fill: { color: PANEL }, line: { color: PANEL, width: 0 } });
  s.addText([{ text: "Below s0 ≈ 130 mm almost nothing is caught, whatever the defect's size. ",
    options: { bold: true, color: INK } },
    { text: "Light from there mostly misses today's 150 mm plane, so training cannot fix it; the " +
      "setup can. That is what the setup sweep measures.", options: { color: INK } }],
    { x: M + 0.25, y: 5.85, w: CW - 0.5, h: 1.0, fontSize: 15, valign: "middle", isTextBox: true });
  s.addNotes("Prelim run on L5_prelim_689da5 (analysis_parameters.png; values read from its bins, " +
    "trained model at the 5% false-alarm threshold). Correlation with s0: crack r = 0.68, dent 0.77; " +
    "depth barely matters (README finding 25). Wear falls as the cut tilts away from the plane and " +
    "rises with depth (0.44 -> 0.94); shell sits at s0 147-154 and is caught 99% of the time.");
}

// ---------------------------------------------------------------- 9. detection rules
{
  const s = pres.addSlide();
  header(s, "DETECTION ONLY", "Is this rail outside the intact bounds?");
  const cw = (CW - 0.3) / 2;
  card(s, M, 1.5, cw, 1.75, "Total barcode difference", ["d = 1 − cos(barcode, mean intact barcode)",
    "One number for the whole pattern; what training optimises."], ACC, { space: 4 });
  card(s, M + cw + 0.3, 1.5, cw, 1.75, "Few-detector out-of-bounds", ["z = max over detectors of " +
    "|b − μ| / σ (intact mean and spread)", "Flags a rail if any one detector leaves its intact range."],
    AMBER, { space: 4 });
  s.addText([{ text: "Scored for every model and setup:  ", options: { bold: true } },
    { text: "AUC  ·  defects caught at 5% false alarms  ·  balanced accuracy  ·  barcode distance " +
      "(defect ÷ alarm line)" }],
    { x: M, y: 3.38, w: CW, h: 0.4, fontSize: 14, color: INK, margin: 0, isTextBox: true });
  placeholder(s, M, 3.95, CW, 2.95, "detect_compare.png",
    "data/generated/2026-11-05/  (train_detect.py --n-det 4 8 12 16)",
    "AUC per rule, recall per class, recall vs position, barcode distance — 4, 8, 12, 16 detectors");
  s.addNotes("Both rules need only intact rails to set the alarm line (95th percentile of the " +
    "validation intact rails), no defect examples at scoring time. The detect+classify model from " +
    "run_stage.py is scored beside the detection-only ones, so the effect of dropping classification " +
    "is measured.");
}

// ---------------------------------------------------------------- 10. sweep design
{
  const s = pres.addSlide();
  header(s, "SETUP SWEEP", "Choosing the setup without training every candidate");
  const hdr = { bold: true, color: WHITE, fill: { color: INK }, fontSize: 14 };
  const rows = [
    [{ text: "Dial", options: hdr }, { text: "Values", options: hdr }, { text: "Cost", options: hdr }],
    ["Horn angle θ", "45°, 55°, 65°", "a new solve"],
    ["Plane height H", "100, 150, 200 mm", "a new solve"],
    ["Plane's far edge", "0.35 to 1.1 × the crown's mirror point", "free: a crop"],
    ["Position at fixed width", "150 or 265 mm wide, 50 mm steps", "free: a crop"]];
  s.addTable(rows, { x: M, y: 1.5, w: 7.3, colW: [2.2, 3.3, 1.8], fontSize: 14, color: INK,
    fontFace: "Calibri", border: { type: "solid", pt: 0.75, color: LINE }, rowH: 0.5,
    fill: { color: WHITE }, valign: "middle" });
  s.addText("Horn distance follows the plane height (its lowest edge stays level with the plane). " +
    "9 solved placements × ~30 crops each.", { x: M, y: 4.15, w: 7.3, h: 0.7, fontSize: 13,
    color: MUTE, italic: true, margin: 0, isTextBox: true });
  const sx = 8.2, sw = W - M - sx;
  stat(s, sx, 1.45, sw, "4,800", "rails per placement: 6 defect types × 7 positions (s 110–170 mm) " +
    "× 100, plus 600 intact", ACC, 40);
  card(s, sx, 3.3, sw, 2.0, "A controlled experiment", ["Sizes spread evenly over each range " +
    "(Latin hypercube)", "The same rails in the same poses for every setup",
    "Each defect's exact signal stored"], ACC, { bullets: true, space: 3 });
  card(s, M, 5.15, 7.3, 1.75, "How each setup is scored", ["No metasurface, 8 detectors, both " +
    "detection rules; 5-fold split by rail, the same folds for every setup."], INK, { space: 4 });
  card(s, sx, 5.5, sw, 1.4, "Run so far", ["θ 45°, 55°, 65° at H 150 (88% of rails);",
    "H 100 and 200 next"], GOOD, { space: 2 });
  s.addNotes("setup_sweep.py + sweep_eval.py, NEXT_PHASE.md. Crop families: 'extend' moves the far " +
    "edge out from +75 mm; 'slide' moves a fixed-width aperture along x. Bright-field crops (the " +
    "crown's mirror point inside the plane) are marked in the figures. The ideal-processor ceiling " +
    "came out ~1.0 everywhere, too loose to prune, so setups are ranked on the detector-only score.");
}

// ---------------------------------------------------------------- 11. sweep results
{
  const s = pres.addSlide();
  header(s, "SETUP SWEEP RESULTS", "A 45° horn and a wider plane: AUC 0.89 → 0.95");
  const ih = 4.6, iw = ih * (3096 / 1483);
  s.addImage({ path: SWEEP + "/sweep_05_detection.png", x: M, y: 1.4, w: iw, h: ih,
    altText: "Every setup ranked by detection AUC, recall and accuracy at 5% false alarms, barcode distance" });
  const sx = M + iw + 0.3, sw = W - M - sx;
  stat(s, sx, 1.35, sw, "0.89 → 0.95", "detection AUC, total barcode difference: today → best", ACC, 28);
  stat(s, sx, 2.95, sw, "74% → 86%", "defects caught at 5% false alarms", GOOD, 28);
  stat(s, sx, 4.55, sw, "0.94 → 0.96", "few-detector out-of-bounds AUC; it beats total difference " +
    "in every setup", ACC, 28);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.15, w: CW, h: 0.85, rectRadius: 0.08,
    fill: { color: PANEL }, line: { color: PANEL, width: 0 } });
  s.addText([{ text: "Horn angle matters most:  ", options: { bold: true, color: ACC } },
    { text: "45° beats 55° beats 65° at nearly every plane width. Best: θ 45°, plane x −135 to +75 mm. " +
      "Untrained, no metasurface, 8 detectors; 3,675 defects and 525 intact rails per setup.",
      options: { color: INK } }],
    { x: M + 0.25, y: 6.17, w: CW - 0.5, h: 0.81, fontSize: 14, valign: "middle", isTextBox: true });
  s.addNotes("sweep_F1_f5be4bbb (commit dbde7d8, geometry 689da5): θ 45/55/65° at H 150, 42-43 of 48 " +
    "shards each. Scores are out-of-fold; the 8 detectors are picked inside each training fold by a " +
    "Fisher x redundancy rule (a stand-in for training + pruning), so these are untrained. Today = " +
    "θ 55°, x −75..75: cos8 AUC 0.891, recall 0.74; best = θ 45°, x −135..75: 0.949, 0.86. The top " +
    "five setups are all 45° and within 0.02 of each other (AUC standard error ~0.005), so treat " +
    "them as tied. The all-detector out-of-bounds rule reaches 0.97-0.99 almost everywhere and " +
    "does not separate setups. The trained linear detector (purple squares, 0.70-0.77) loses to " +
    "both untrained rules: defects push the barcode in different directions, which one linear " +
    "boundary cannot follow. The learning curve is flat past half the data, so the missing 12% of " +
    "rails does not change the ranking.");
}

// ---------------------------------------------------------------- 12. where each setup looks
{
  const s = pres.addSlide();
  header(s, "SETUP SWEEP RESULTS", "The wider plane wins the crown and loses the gauge face");
  s.addImage({ path: SWEEP + "/sweep_07_detection_cells.png", x: M, y: 1.4, w: CW,
    h: CW / (3033 / 660), altText: "Defects caught by type and position, today vs the best setup" });
  const fh = 2.75, fw = fh * (772 / 704), fy = 4.2;
  s.addImage({ path: SWEEP + "/sweep_06a_far_edge.png", x: M, y: fy, w: fw, h: fh,
    altText: "Detection AUC vs the plane's far edge for each horn angle" });
  const cx = M + fw + 0.25, cw2 = (W - M - cx - 0.25) / 2;
  card(s, cx, fy, cw2, fh, "Gained: the crown side (s 110–130)", ["Oblique cracks 19–31% → 47–97%",
    "Transverse cracks at s 110: 28% → 78%", "Dents at s 110: 24% → 69%",
    "Longitudinal cracks: 91% or better everywhere, in both"], GOOD,
    { bullets: true, space: 5, bodySize: 15 });
  card(s, cx + cw2 + 0.25, fy, cw2, fh, "Lost: the gauge face (s 170)", ["Shells 18% → 6%",
    "Transverse cracks 56% → 27%; oblique 36% → 19%", "The near edge stayed at +75 mm: " +
    "widening it is the next test"], BAD, { bullets: true, space: 5, bodySize: 15 });
  s.addNotes("Top: defects caught at 5% false alarms per defect type x position, today (θ 55°, " +
    "x −75..75) vs the best setup (θ 45°, x −135..75), both rules; the card numbers are the total " +
    "barcode difference rule. Bottom left: AUC vs how far the plane reaches toward the crown's " +
    "mirror point, near edge fixed at +75 mm. AUC rises to ~0.9 of the mirror point at 45° and " +
    "55°; taking the glint inside does not help at 45°. 65° stays near today's level, so the " +
    "design study's 65° pick (a field-level measure) does not survive detection scoring. Only " +
    "H 150 has run, so the horn-angle x height grid is one row (left out).");
}

// ---------------------------------------------------------------- 13. next steps
{
  const s = pres.addSlide();
  header(s, "NEXT STEPS", "Let the data pick the setup, then train the finalists");
  const items = [
    ["Finish the sweep", "H 100 and 200; widen the plane's near edge to win back the gauge face."],
    ["Train the finalists", "θ 45°, plane x −135 to +75 mm (and today's width slid to −125 to +25); 4 to 16 detectors."],
    ["Where it fails", "Recall by defect type × position × size, not one overall number."],
    ["Fixable or not?", "Label each miss by its physical signal strength: weak signals cannot be trained away."],
    ["How much data", "Learning curves per cell: spend the next generation where results still climb."],
    ["Trust the labels", "Cracks agree with full-wave only 46–67%: flag them; FDTD rung 3 next."]];
  const cw = (CW - 2 * 0.3) / 3, ch = 2.45;
  items.forEach(([h, t], i) => {
    const x = M + (i % 3) * (cw + 0.3), y = 1.5 + Math.floor(i / 3) * (ch + 0.3);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, rectRadius: 0.08,
      fill: { color: PANEL }, line: { color: PANEL, width: 0 } });
    dot(s, x + 0.25, y + 0.25, 0.46, i + 1, i < 2 ? ACC : GOOD);
    s.addText([{ text: h, options: { bold: true, fontSize: 17, color: INK, breakLine: true } },
      { text: t, options: { fontSize: 14, color: INK } }],
      { x: x + 0.25, y: y + 0.85, w: cw - 0.5, h: ch - 1.0, valign: "top", margin: 0,
        paraSpaceAfter: 5, isTextBox: true });
  });
  s.addNotes("Items 3-6 are the data-centric track (NEXT_PHASE.md section 4: D1 stratified " +
    "evaluation, D2 difficulty labels from the stored exact defect signal, D3 learning curves, " +
    "D4 simulation-label fidelity from finding 34).");
}

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote " + f));
