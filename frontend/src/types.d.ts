declare module "plotly.js" {
  const plotly: any;
  export default plotly;
  export type Data = any;
  export type Layout = any;
  export type Config = any;
}

declare module "*.css" {}

declare module "react-plotly.js" {
  import { Component } from "react";
  import Plotly from "plotly.js";

  interface PlotParams {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    style?: React.CSSProperties;
    className?: string;
    onInitialized?: (figure: any, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: any, graphDiv: HTMLElement) => void;
  }

  class Plot extends Component<PlotParams> {}
  export default Plot;
}
