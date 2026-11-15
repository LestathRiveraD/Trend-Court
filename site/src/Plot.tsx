import { useEffect, useRef, useState } from "react";
import type { Config, Data, Layout } from "plotly.js-dist-min";

interface PlotProps {
  data: Data[];
  layout?: Partial<Layout>;
  label: string;
}

const CONFIG: Partial<Config> = { responsive: true, displayModeBar: false };

/**
 * Thin wrapper over the bundled Plotly (`plotly.js-dist-min`, never a CDN).
 * Plotly is imported lazily so the shell paints before the ~4.6 MB chunk arrives.
 */
export default function Plot({ data, layout, label }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let cancelled = false;
    let plotly: typeof import("plotly.js-dist-min").default | null = null;
    import("plotly.js-dist-min")
      .then(async ({ default: mod }) => {
        if (cancelled) return;
        plotly = mod;
        await mod.react(el, data, { autosize: true, ...layout }, CONFIG);
      })
      .catch((err: Error) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
      plotly?.purge(el);
    };
  }, [data, layout]);

  if (error) return <p role="alert" className="error">Chart failed to load: {error}</p>;
  return <div ref={ref} className="plot" role="img" aria-label={label} />;
}
