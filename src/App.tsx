import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Brain,
  SlidersHorizontal,
} from "lucide-react";

// ─────────────────────────────────────────────────────────────────────────────
// Minimal local UI primitives (no shadcn/ui needed)
// ─────────────────────────────────────────────────────────────────────────────
function cx(...xs: Array<string | false | null | undefined>) {
  return xs.filter(Boolean).join(" ");
}
const Button: React.FC<
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: "default" | "secondary" | "outline";
  }
> = ({ variant = "default", className = "", ...props }) => (
  <button
    className={cx(
      "inline-flex items-center gap-2 px-3 py-2 rounded-2xl text-sm transition-colors",
      variant === "secondary" && "bg-slate-200 hover:bg-slate-300",
      variant === "outline" &&
        "border border-slate-300 bg-white hover:bg-slate-50",
      variant === "default" && "bg-indigo-600 text-white hover:bg-indigo-700",
      className
    )}
    {...props}
  />
);
const Card: React.FC<{ className?: string; children: React.ReactNode }> = ({
  className = "",
  children,
}) => (
  <div className={cx("bg-white rounded-2xl shadow", className)}>{children}</div>
);
const CardHeader: React.FC<{
  className?: string;
  children: React.ReactNode;
}> = ({ className = "", children }) => (
  <div className={cx("px-4 pt-4", className)}>{children}</div>
);
const CardTitle: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <h2 className="text-lg font-semibold tracking-tight">{children}</h2>
);
const CardContent: React.FC<{
  className?: string;
  children: React.ReactNode;
}> = ({ className = "", children }) => (
  <div className={cx("px-4 pb-4", className)}>{children}</div>
);

// ─────────────────────────────────────────────────────────────────────────────
// Math utils
// ─────────────────────────────────────────────────────────────────────────────
function randn() {
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
function seededRandom(seed: number) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}

// Activations
const act = {
  tanh: {
    f: (x: number) => Math.tanh(x),
    df: (x: number) => 1 - Math.tanh(x) ** 2,
    name: "tanh",
  },
  relu: {
    f: (x: number) => (x > 0 ? x : 0),
    df: (x: number) => (x > 0 ? 1 : 0),
    name: "ReLU",
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Dataset factory (2D→target) w/ classification & regression
// ─────────────────────────────────────────────────────────────────────────────
function makeDatasets() {
  const gen = {
    moons: (n = 200, noise = 0.15, seed = 7) => {
      const rng = seededRandom(seed);
      const pts: { x: number; y: number; t: number }[] = [];
      for (let i = 0; i < n; i++) {
        const th = (i / n) * Math.PI;
        const r1 = 1 + (rng() - 0.5) * noise * 2;
        pts.push({
          x: r1 * Math.cos(th) - 0.2 + randn() * noise * 0.15,
          y: r1 * Math.sin(th) + 0.1 + randn() * noise * 0.15,
          t: 0,
        });
      }
      for (let i = 0; i < n; i++) {
        const th = (i / n) * Math.PI;
        const r2 = 1 + (rng() - 0.5) * noise * 2;
        pts.push({
          x: 1.0 + r2 * Math.cos(th) + 0.2 + randn() * noise * 0.15,
          y: -r2 * Math.sin(th) - 0.1 + randn() * noise * 0.15,
          t: 1,
        });
      }
      return {
        name: "Moons (classification)",
        task: "classification" as const,
        points: pts,
      };
    },
    circles: (n = 200, noise = 0.1) => {
      const pts: { x: number; y: number; t: number }[] = [];
      for (let i = 0; i < n; i++) {
        const th = (2 * Math.PI * i) / n;
        const r = 0.6 + randn() * noise;
        pts.push({ x: r * Math.cos(th), y: r * Math.sin(th), t: 0 });
      }
      for (let i = 0; i < n; i++) {
        const th = (2 * Math.PI * i) / n;
        const r = 1.3 + randn() * noise;
        pts.push({ x: r * Math.cos(th), y: r * Math.sin(th), t: 1 });
      }
      return {
        name: "Concentric circles (classification)",
        task: "classification" as const,
        points: pts,
      };
    },
    xor: (n = 400, noise = 0.15, seed = 7) => {
      const rng = seededRandom(seed);
      const pts: { x: number; y: number; t: number }[] = [];
      for (let i = 0; i < n; i++) {
        const x = (rng() * 2 - 1) * 2;
        const y = (rng() * 2 - 1) * 2;
        const t = x > 0 !== y > 0 ? 1 : 0;
        pts.push({ x: x + randn() * noise, y: y + randn() * noise, t });
      }
      return {
        name: "XOR checkerboard (classification)",
        task: "classification" as const,
        points: pts,
      };
    },
    spirals: (n = 200, noise = 0.1) => {
      const pts: { x: number; y: number; t: number }[] = [];
      for (let i = 0; i < n; i++) {
        const t = (i / n) * 4 * Math.PI;
        const r = 0.1 + t * 0.08;
        pts.push({
          x: r * Math.cos(t) + randn() * noise,
          y: r * Math.sin(t) + randn() * noise,
          t: 0,
        });
      }
      for (let i = 0; i < n; i++) {
        const t = (i / n) * 4 * Math.PI + Math.PI;
        const r = 0.1 + (t - Math.PI) * 0.08;
        pts.push({
          x: r * Math.cos(t) + randn() * noise,
          y: r * Math.sin(t) + randn() * noise,
          t: 1,
        });
      }
      return {
        name: "Two spirals (classification)",
        task: "classification" as const,
        points: pts,
      };
    },
    sinc: (n = 600, noise = 0.05, seed = 7) => {
      const rng = seededRandom(seed);
      const pts: { x: number; y: number; t: number }[] = [];
      for (let i = 0; i < n; i++) {
        const x = (rng() * 2 - 1) * 3;
        const y = (rng() * 2 - 1) * 3;
        const r = Math.hypot(x, y) + 1e-6;
        pts.push({ x, y, t: Math.sin(r) / r + randn() * noise });
      }
      return {
        name: "Sinc ripple (regression)",
        task: "regression" as const,
        points: pts,
      };
    },
    plane: (n = 600, noise = 0.05, seed = 7) => {
      const rng = seededRandom(seed);
      const pts: { x: number; y: number; t: number }[] = [];
      for (let i = 0; i < n; i++) {
        const x = (rng() * 2 - 1) * 2;
        const y = (rng() * 2 - 1) * 2;
        pts.push({ x, y, t: 0.6 * x - 0.3 * y + 0.2 + randn() * noise });
      }
      return {
        name: "Noisy plane (regression)",
        task: "regression" as const,
        points: pts,
      };
    },
  };
  return gen;
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic MLP (2D→1) with variable hidden layers; full‑batch GD + grad probe
// ─────────────────────────────────────────────────────────────────────────────
function useMLP({
  layers = [6],
  activation = "tanh",
  task = "classification" as "classification" | "regression",
}) {
  const H = layers;
  const [W, setW] = useState<number[][][]>(() => initWeights(H));
  const [B, setB] = useState<number[][]>(() => initBiases(H));
  const actFun = act[activation as keyof typeof act];

  function initWeights(h: number[]) {
    const sizes = [2, ...h, 1];
    return sizes.slice(0, -1).map((_, l) => {
      const inS = sizes[l],
        outS = sizes[l + 1];
      const scale = 1 / Math.sqrt(inS);
      return Array.from({ length: inS }, () =>
        Array.from({ length: outS }, () => randn() * scale)
      );
    });
  }
  function initBiases(h: number[]) {
    const sizes = [2, ...h, 1];
    return sizes
      .slice(1)
      .map((outS) => Array.from({ length: outS }, () => randn() * 0.05));
  }
  function reinit(newLayers = H) {
    setW(initWeights(newLayers));
    setB(initBiases(newLayers));
  }

  function forward(x: number, y: number) {
    const sizes = [2, ...H, 1];
    const a: number[][] = [[x, y]],
      z: number[][] = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const inS = sizes[l],
        outS = sizes[l + 1];
      const zl = Array(outS).fill(0);
      for (let j = 0; j < outS; j++) {
        let s = B[l][j];
        for (let i = 0; i < inS; i++) s += a[l][i] * W[l][i][j];
        zl[j] = s;
      }
      z.push(zl);
      const isOut = l === sizes.length - 2;
      if (isOut)
        a.push(
          zl.map((v) =>
            task === "classification" ? 1 / (1 + Math.exp(-v)) : v
          )
        );
      else a.push(zl.map(actFun.f));
    }
    return { a, z, yhat: a[a.length - 1][0] };
  }
  function predictBatch(XY: { x: number; y: number }[]) {
    return XY.map((p) => forward(p.x, p.y).yhat);
  }

  function computeBatchGrads(XY: { x: number; y: number }[], yTrue: number[]) {
    const sizes = [2, ...H, 1];
    const gW = W.map((layer) => layer.map((row) => row.map(() => 0)));
    const gB = B.map((arr) => arr.map(() => 0));
    let loss = 0;

    for (let n = 0; n < XY.length; n++) {
      const { x, y } = XY[n];
      const t = yTrue[n];
      const { a, z, yhat } = forward(x, y);
      let deltaL: number[];
      if (task === "classification") {
        loss += -(
          t * Math.log(yhat + 1e-8) +
          (1 - t) * Math.log(1 - yhat + 1e-8)
        );
        deltaL = [yhat - t];
      } else {
        const e = yhat - t;
        loss += 0.5 * e * e;
        deltaL = [e];
      }

      const deltas: number[][] = Array(
        sizes.length - 1
      ) as unknown as number[][];
      deltas[deltas.length - 1] = deltaL;

      for (let l = sizes.length - 3; l >= 0; l--) {
        const outS = sizes[l + 1],
          nextS = sizes[l + 2];
        const dl = Array(outS).fill(0);
        for (let i = 0; i < outS; i++) {
          let s = 0;
          for (let j = 0; j < nextS; j++)
            s += deltas[l + 1][j] * W[l + 1][i][j];
          dl[i] = s * actFun.df(z[l][i]);
        }
        deltas[l] = dl;
      }

      for (let l = 0; l < sizes.length - 1; l++) {
        const inS = sizes[l],
          outS = sizes[l + 1];
        for (let j = 0; j < outS; j++) {
          gB[l][j] += deltas[l][j];
          for (let i = 0; i < inS; i++) gW[l][i][j] += a[l][i] * deltas[l][j];
        }
      }
    }
    const n = XY.length;
    loss /= n;
    const gWavg = gW.map((layer) => layer.map((row) => row.map((v) => v / n)));
    const gBavg = gB.map((arr) => arr.map((v) => v / n));
    return { gW: gWavg, gB: gBavg, loss };
  }

  function step(XY: { x: number; y: number }[], yTrue: number[], lr = 0.1) {
    const { gW, gB, loss } = computeBatchGrads(XY, yTrue);
    const scale = lr;
    setW((prev) =>
      prev.map((layer, l) =>
        layer.map((row, i) => row.map((w, j) => w - scale * gW[l][i][j]))
      )
    );
    setB((prev) =>
      prev.map((arr, l) => arr.map((bv, j) => bv - scale * gB[l][j]))
    );
    return loss;
  }

  return { W, B, forward, predictBatch, step, reinit, computeBatchGrads };
}

// ─────────────────────────────────────────────────────────────────────────────
// Visualization components
// ─────────────────────────────────────────────────────────────────────────────
function DecisionCanvas({
  data,
  predict,
  task = "classification",
  width = 520,
  height = 380,
  padding = 24,
}: {
  data: { x: number; y: number; t: number }[];
  predict: (xy: { x: number; y: number }[]) => number[];
  task?: "classification" | "regression";
  width?: number;
  height?: number;
  padding?: number;
}) {
  const xs = data.map((p) => p.x),
    ys = data.map((p) => p.y),
    ts = data.map((p) => p.t);
  const xmin = Math.min(...xs) - 0.5,
    xmax = Math.max(...xs) + 0.5;
  const ymin = Math.min(...ys) - 0.5,
    ymax = Math.max(...ys) + 0.5;
  const tmin = Math.min(...ts),
    tmax = Math.max(...ts);
  const x2sx = (x: number) =>
    padding + ((x - xmin) / (xmax - xmin)) * (width - 2 * padding);
  const y2sy = (y: number) =>
    height - padding - ((y - ymin) / (ymax - ymin)) * (height - 2 * padding);

  const grid: { gx: number; gy: number; p: number }[] = [];
  const nx = 90,
    ny = 60;
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++) {
      const gx = xmin + ((i + 0.5) * (xmax - xmin)) / nx;
      const gy = ymin + ((j + 0.5) * (ymax - ymin)) / ny;
      grid.push({ gx, gy, p: predict([{ x: gx, y: gy }])[0] });
    }
  function colorFor(p: number) {
    if (task === "classification")
      return `rgba(${Math.round(255 * p)}, ${Math.round(
        255 * (1 - p)
      )}, 200, 0.35)`;
    const pn = (p - tmin) / (tmax - tmin + 1e-9);
    return `rgba(${Math.round(255 * pn)}, ${Math.round(
      255 * (1 - pn)
    )}, 200, 0.35)`;
  }
  return (
    <svg width={width} height={height} className="rounded-2xl shadow bg-white">
      {grid.map((g, idx) => (
        <rect
          key={idx}
          x={x2sx(g.gx) - (width - 2 * padding) / 90 / 2}
          y={y2sy(g.gy) - (height - 2 * padding) / 60 / 2}
          width={(width - 2 * padding) / 90}
          height={(height - 2 * padding) / 60}
          fill={colorFor(g.p)}
        />
      ))}
      <line
        x1={padding}
        y1={height - padding}
        x2={width - padding}
        y2={height - padding}
        stroke="#222"
        strokeWidth={1}
      />
      <line
        x1={padding}
        y1={height - padding}
        x2={padding}
        y2={padding}
        stroke="#222"
        strokeWidth={1}
      />
      {data.map((p, i) => (
        <circle
          key={i}
          cx={x2sx(p.x)}
          cy={y2sy(p.y)}
          r={4}
          fill={
            task === "classification"
              ? p.t === 1
                ? "#1f77b4"
                : "#d62728"
              : "#111827"
          }
          opacity={0.95}
        />
      ))}
    </svg>
  );
}

function NetworkDiagram({
  W,
  B,
  activation,
  lastForward,
  overlayMode = "none",
  overlayData = null,
}: {
  W: number[][][];
  B: number[][];
  activation: string;
  lastForward: any;
  overlayMode?: "none" | "grad";
  overlayData?: { gW: number[][][]; gB: number[][] } | null;
}) {
  const sizes = [2, ...W.map((layer) => layer[0].length)];
  const hiddenCount = sizes.length - 2;
  const width = 100 + 120 * (hiddenCount + 1);
  const layerX = Array.from(
    { length: hiddenCount + 2 },
    (_, i) => 50 + i * 120
  );
  const layerY = (n: number) =>
    Array.from({ length: n }, (_, i) => 30 + (i + 1) * (180 / (n + 1)));
  const positions = [layerY(2), ...sizes.slice(1).map(layerY)];

  const maxPerLayer = useMemo(() => {
    if (!overlayData || !overlayData.gW) return [] as number[];
    return overlayData.gW.map((layer) => {
      let m = 0;
      for (let i = 0; i < layer.length; i++)
        for (let j = 0; j < layer[i].length; j++)
          m = Math.max(m, Math.abs(layer[i][j]));
      return m || 1;
    });
  }, [overlayData]);

  const baseEdge = (
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    w: number
  ) => (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke={w >= 0 ? "#2563eb" : "#dc2626"}
      strokeOpacity={0.7}
      strokeWidth={Math.min(6, Math.max(1, Math.abs(w) * 1.5))}
    />
  );

  const overlayEdge = (
    l: number,
    i: number,
    j: number,
    x1: number,
    y1: number,
    x2: number,
    y2: number
  ) => {
    if (!overlayData || !overlayData.gW || overlayMode === "none") return null;
    const g = overlayData.gW[l][i][j];
    const a = Math.min(1, Math.abs(g) / (maxPerLayer[l] || 1));
    const color = g >= 0 ? "#22c55e" : "#ef4444"; // green (+), red (-)
    const width = 1 + 6 * a;
    return (
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={color}
        strokeOpacity={0.9}
        strokeWidth={width}
        strokeLinecap="round"
      />
    );
  };

  return (
    <svg width={width} height={240} className="bg-white rounded-2xl shadow">
      {W.map((Wl, l) => (
        <g key={`L${l}`}>
          {Wl.map((row, i) =>
            row.map((wij, j) => (
              <g key={`E${l}-${i}-${j}`}>
                {baseEdge(
                  layerX[l],
                  positions[l][i],
                  layerX[l + 1],
                  positions[l + 1][j],
                  wij
                )}
                {overlayEdge(
                  l,
                  i,
                  j,
                  layerX[l],
                  positions[l][i],
                  layerX[l + 1],
                  positions[l + 1][j]
                )}
              </g>
            ))
          )}
        </g>
      ))}
      {positions.map((ys, l) =>
        ys.map((y, i) => (
          <motion.circle
            key={`N${l}-${i}`}
            cx={layerX[l]}
            cy={y}
            r={l === 0 ? 12 : 14}
            fill="#111827"
            animate={{
              scale: lastForward
                ? 1 + 0.06 * Math.tanh(Math.abs(lastForward.a?.[l]?.[i] ?? 0))
                : 1,
            }}
            transition={{ type: "spring", stiffness: 120, damping: 10 }}
          />
        ))
      )}
      <text
        x={layerX[0]}
        y={16}
        textAnchor="middle"
        fontSize={12}
        fill="#374151"
      >
        Input (x,y)
      </text>
      <text
        x={layerX[layerX.length - 2]}
        y={16}
        textAnchor="middle"
        fontSize={12}
        fill="#374151"
      >
        Hidden ×{hiddenCount} [{activation}]
      </text>
      <text
        x={layerX[layerX.length - 1]}
        y={16}
        textAnchor="middle"
        fontSize={12}
        fill="#374151"
      >
        Output
      </text>
    </svg>
  );
}

function LossChart({
  history,
  width = 360,
  height = 220,
}: {
  history: { step: number; loss: number }[];
  width?: number;
  height?: number;
}) {
  if (history.length === 0)
    return (
      <div className="w-[360px] h-[220px] bg-white rounded-2xl shadow grid place-items-center text-sm text-gray-500">
        No training yet
      </div>
    );
  const maxN = 200;
  const hist = history.slice(-maxN);
  const xs = hist.map((d) => d.step);
  const ys = hist.map((d) => d.loss);
  const xmin = Math.min(...xs),
    xmax = Math.max(...xs),
    ymin = Math.min(...ys),
    ymax = Math.max(...ys);
  const pad = 24;
  const x2sx = (x: number) =>
    pad + ((x - xmin) / (xmax - xmin || 1)) * (width - 2 * pad);
  const y2sy = (y: number) =>
    height - pad - ((y - ymin) / (ymax - ymin || 1)) * (height - 2 * pad);
  return (
    <svg width={width} height={height} className="bg-white rounded-2xl shadow">
      <polyline
        fill="none"
        stroke="#10b981"
        strokeWidth={2}
        points={hist.map((d) => `${x2sx(d.step)},${y2sy(d.loss)}`).join(" ")}
      />
      <text x={pad} y={16} fontSize={12} fill="#374151">
        Loss
      </text>
      <line
        x1={pad}
        y1={height - pad}
        x2={width - pad}
        y2={height - pad}
        stroke="#9ca3af"
        strokeWidth={1}
      />
      <line
        x1={pad}
        y1={pad}
        x2={pad}
        y2={height - pad}
        stroke="#9ca3af"
        strokeWidth={1}
      />
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  // Datasets
  const datasets = useMemo(() => makeDatasets(), []);
  const [dsKey, setDsKey] =
    useState<keyof ReturnType<typeof makeDatasets>>("moons");
  const [seed, setSeed] = useState(7);
  const [noise, setNoise] = useState(0.15);
  const [nPerClass, setNPerClass] = useState(120);

  const dsObj = useMemo(() => {
    const baseN =
      dsKey === "sinc" || dsKey === "plane" ? nPerClass * 2 : nPerClass;
    return datasets[dsKey](baseN, noise, seed);
  }, [datasets, dsKey, nPerClass, noise, seed]);

  const data = dsObj.points as { x: number; y: number; t: number }[];
  const task = dsObj.task as "classification" | "regression";
  const XY = useMemo(() => data.map(({ x, y }) => ({ x, y })), [data]);
  const yTrue = useMemo(() => data.map(({ t }) => t), [data]);

  // Network
  const [numLayers, setNumLayers] = useState(2);
  const [layerSizes, setLayerSizes] = useState([6, 6, 6, 6]);
  const activeLayers = useMemo(
    () => layerSizes.slice(0, numLayers),
    [layerSizes, numLayers]
  );
  const [activation, setActivation] = useState<keyof typeof act>("tanh");

  const { W, B, predictBatch, step, reinit, forward, computeBatchGrads } =
    useMLP({ layers: activeLayers, activation, task });

  // Training state
  const [lr, setLr] = useState(0.2);
  const [running, setRunning] = useState(false);
  const [iter, setIter] = useState(0);
  const [lossHist, setLossHist] = useState<{ step: number; loss: number }[]>(
    []
  );
  const [lastFwd, setLastFwd] = useState<any>(null);

  // Overlay state
  const [overlayMode, setOverlayMode] = useState<"none" | "grad">("none");
  const [overlayData, setOverlayData] = useState<{
    gW: number[][][];
    gB: number[][];
  } | null>(null);

  useEffect(() => {
    let raf: number | undefined;
    function tick() {
      const L = step(XY, yTrue, lr);
      setIter((k) => k + 1);
      setLossHist((h) => [...h, { step: (h.at(-1)?.step ?? 0) + 1, loss: L }]);
      const i = Math.floor(Math.random() * XY.length);
      setLastFwd(forward(XY[i].x, XY[i].y));
      if (overlayMode !== "none") setOverlayData(computeBatchGrads(XY, yTrue));
      raf = requestAnimationFrame(tick);
    }
    if (running) raf = requestAnimationFrame(tick);
    return () => raf && cancelAnimationFrame(raf);
  }, [running, lr, step, XY, yTrue, forward, overlayMode, computeBatchGrads]);

  function doStep() {
    const L = step(XY, yTrue, lr);
    setIter((k) => k + 1);
    setLossHist((h) => [...h, { step: (h.at(-1)?.step ?? 0) + 1, loss: L }]);
    const i = Math.floor(Math.random() * XY.length);
    setLastFwd(forward(XY[i].x, XY[i].y));
    if (overlayMode !== "none") setOverlayData(computeBatchGrads(XY, yTrue));
  }

  function resetAll() {
    setRunning(false);
    setIter(0);
    setLossHist([]);
    setSeed((s) => s + 1);
    reinit(activeLayers);
    setOverlayData(null);
  }

  const preds = useMemo(() => predictBatch(XY), [XY, predictBatch]);
  const metric = useMemo(() => {
    if (!preds.length)
      return {
        label: task === "classification" ? "Accuracy" : "RMSE",
        value: 0,
      };
    if (task === "classification") {
      let correct = 0;
      for (let i = 0; i < preds.length; i++)
        correct += preds[i] >= 0.5 === (yTrue[i] === 1) ? 1 : 0;
      return { label: "Accuracy", value: correct / preds.length };
    }
    let mse = 0;
    for (let i = 0; i < preds.length; i++) {
      const e = preds[i] - yTrue[i];
      mse += e * e;
    }
    return { label: "RMSE", value: Math.sqrt(mse / preds.length) };
  }, [preds, yTrue, task]);

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-50 to-slate-200 p-6">
      <div className="max-w-7xl mx-auto">
        <header className="flex items-center gap-3 mb-6">
          <div className="h-10 w-10 grid place-items-center rounded-2xl bg-indigo-600 text-white shadow">
            <Brain className="h-6 w-6" />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">
            NeuroStudio — Interactive Neural Network Tutor
          </h1>
        </header>

        <div className="grid md:grid-cols-2 gap-6">
          <Card className="overflow-hidden">
            <CardHeader className="pb-2">
              <CardTitle>Decision Field & Data ({dsObj.name})</CardTitle>
            </CardHeader>
            <CardContent>
              <DecisionCanvas data={data} predict={predictBatch} task={task} />
              <div className="mt-3 text-sm text-slate-600">
                {metric.label}:{" "}
                <span className="font-semibold text-slate-800">
                  {task === "classification"
                    ? (metric.value * 100).toFixed(1) + "%"
                    : metric.value.toFixed(3)}
                </span>
                <span className="ml-4">
                  Iter: <span className="font-mono">{iter}</span>
                </span>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-6">
            <Card>
              <CardHeader className="pb-1">
                <CardTitle>Network Diagram</CardTitle>
              </CardHeader>
              <CardContent>
                <NetworkDiagram
                  W={W}
                  B={B}
                  activation={activation}
                  lastForward={lastFwd}
                  overlayMode={overlayMode}
                  overlayData={overlayData}
                />
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-1">
                <CardTitle>Loss</CardTitle>
              </CardHeader>
              <CardContent>
                <LossChart history={lossHist} />
              </CardContent>
            </Card>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6 mt-6">
          <Card>
            <CardHeader className="pb-1">
              <CardTitle>Train Controls</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  onClick={() => setRunning((r) => !r)}
                  className="rounded-2xl"
                >
                  {running ? (
                    <Pause className="mr-2 h-4 w-4" />
                  ) : (
                    <Play className="mr-2 h-4 w-4" />
                  )}{" "}
                  {running ? "Pause" : "Run"}
                </Button>
                <Button
                  variant="secondary"
                  onClick={doStep}
                  className="rounded-2xl"
                >
                  <StepForward className="mr-2 h-4 w-4" /> Step
                </Button>
                <Button
                  variant="outline"
                  onClick={resetAll}
                  className="rounded-2xl"
                >
                  <RotateCcw className="mr-2 h-4 w-4" /> Reset
                </Button>
              </div>
              <div className="mt-4 grid grid-cols-7 items-center gap-3">
                <label className="col-span-3 text-sm text-slate-600">
                  Learning rate
                </label>
                <input
                  type="range"
                  min="0.01"
                  max="1"
                  step={0.01}
                  value={lr}
                  onChange={(e) => setLr(parseFloat(e.target.value))}
                  className="col-span-4"
                />
                <div className="col-span-7 text-xs text-slate-500">
                  {lr.toFixed(2)}
                </div>
              </div>
              <div className="mt-4 grid grid-cols-7 items-center gap-3">
                <label className="col-span-3 text-sm text-slate-600">
                  Overlay
                </label>
                <select
                  className="col-span-4 border rounded-xl p-1 bg-white"
                  value={overlayMode}
                  onChange={(e) => setOverlayMode(e.target.value as any)}
                >
                  <option value="none">None</option>
                  <option value="grad">Gradient magnitude (∂L/∂w)</option>
                </select>
                <div className="col-span-7 text-xs text-slate-500">
                  Pause or step, choose an overlay, then click{" "}
                  <strong>Probe grads</strong> to compute and render per‑edge
                  values.
                </div>
                <Button
                  variant="secondary"
                  className="rounded-2xl col-span-7"
                  onClick={() => setOverlayData(computeBatchGrads(XY, yTrue))}
                >
                  Probe grads
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-1">
              <CardTitle>Network Settings</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-7 items-center gap-3">
                <label className="col-span-3 text-sm text-slate-600">
                  Hidden layers
                </label>
                <input
                  type="range"
                  min={1}
                  max={4}
                  step={1}
                  value={numLayers}
                  onChange={(e) => {
                    const n = parseInt(e.target.value);
                    setNumLayers(n);
                    reinit(layerSizes.slice(0, n));
                  }}
                  className="col-span-4"
                />
                <div className="col-span-7 text-xs text-slate-500">
                  {numLayers}
                </div>
              </div>
              {Array.from({ length: numLayers }, (_, idx) => (
                <div
                  key={idx}
                  className="grid grid-cols-7 items-center gap-3 mt-3"
                >
                  <label className="col-span-3 text-sm text-slate-600">
                    Units L{idx + 1}
                  </label>
                  <input
                    type="range"
                    min={2}
                    max={32}
                    step={1}
                    value={layerSizes[idx]}
                    onChange={(e) => {
                      const v = parseInt(e.target.value);
                      const next = [...layerSizes];
                      next[idx] = v;
                      setLayerSizes(next);
                      reinit(next.slice(0, numLayers));
                    }}
                    className="col-span-4"
                  />
                  <div className="col-span-7 text-xs text-slate-500">
                    {layerSizes[idx]}
                  </div>
                </div>
              ))}
              <div className="mt-3 flex items-center gap-2 text-sm">
                <span className="text-slate-600">Activation:</span>
                {(["tanh", "relu"] as const).map((a) => (
                  <button
                    key={a}
                    onClick={() => setActivation(a)}
                    className={cx(
                      "px-3 py-1 rounded-xl border",
                      activation === a
                        ? "bg-indigo-600 text-white"
                        : "bg-white text-slate-700"
                    )}
                  >
                    {a}
                  </button>
                ))}
              </div>
              <p className="mt-3 text-xs text-slate-500 leading-relaxed">
                Output:{" "}
                {task === "classification"
                  ? "sigmoid (binary)"
                  : "identity (regression)"}{" "}
                · Loss: {task === "classification" ? "BCE" : "MSE"}.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-1">
              <CardTitle>Data Settings</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-7 items-center gap-3">
                <label className="col-span-3 text-sm text-slate-600">
                  Dataset
                </label>
                <select
                  className="col-span-4 border rounded-xl p-1 bg-white"
                  value={dsKey}
                  onChange={(e) => setDsKey(e.target.value as any)}
                >
                  <option value="moons">Moons (classification)</option>
                  <option value="circles">
                    Concentric circles (classification)
                  </option>
                  <option value="xor">XOR checkerboard (classification)</option>
                  <option value="spirals">Two spirals (classification)</option>
                  <option value="sinc">Sinc ripple (regression)</option>
                  <option value="plane">Noisy plane (regression)</option>
                </select>
              </div>
              <div className="grid grid-cols-7 items-center gap-3 mt-3">
                <label className="col-span-3 text-sm text-slate-600">
                  Points (per class)
                </label>
                <input
                  type="range"
                  min={30}
                  max={400}
                  step={10}
                  value={nPerClass}
                  onChange={(e) => setNPerClass(parseInt(e.target.value))}
                  className="col-span-4"
                />
                <div className="col-span-7 text-xs text-slate-500">
                  {nPerClass}
                </div>
              </div>
              <div className="grid grid-cols-7 items-center gap-3 mt-3">
                <label className="col-span-3 text-sm text-slate-600">
                  Noise
                </label>
                <input
                  type="range"
                  min={0}
                  max={0.6}
                  step={0.01}
                  value={noise}
                  onChange={(e) =>
                    setNoise(parseFloat((e.target as HTMLInputElement).value))
                  }
                  className="col-span-4"
                />
                <div className="col-span-7 text-xs text-slate-500">
                  {noise.toFixed(2)}
                </div>
              </div>
              <div className="mt-3 flex items-center gap-2">
                <Button
                  variant="secondary"
                  className="rounded-2xl"
                  onClick={() => setSeed((s) => s + 1)}
                >
                  <SlidersHorizontal className="h-4 w-4" /> Reroll data
                </Button>
              </div>
              <p className="mt-3 text-xs text-slate-500">
                Classification points are colored by class; regression points
                are black; background tint encodes predicted value.
              </p>
            </CardContent>
          </Card>
        </div>

        <Card className="mt-6">
          <CardHeader className="pb-1">
            <CardTitle>What’s going on under the hood?</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc ml-5 text-sm text-slate-700 space-y-1">
              <li>
                Model: 2 → {activeLayers.join(" → ")} → 1 MLP. Hidden activation
                selectable; output depends on task.
              </li>
              <li>
                Loss:{" "}
                {task === "classification"
                  ? "Binary cross-entropy"
                  : "Mean squared error"}
                . Optimizer: full-batch gradient descent.
              </li>
              <li>
                Overlay: Probe full-batch <code>∂L/∂w</code> and visualize
                magnitude/sign per edge (green = +, red = −).
              </li>
              <li>
                Decision field shows σ(z) for classification and raw output for
                regression (color normalized).
              </li>
            </ul>
          </CardContent>
        </Card>

        <footer className="text-xs text-slate-500 mt-6">
          Next extensions: minibatches + Adam; multi-class softmax + 3-class
          blobs; hover tooltips with (w, ∂L/∂w, −η∂L/∂w).
        </footer>
      </div>
    </div>
  );
}
