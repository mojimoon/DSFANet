"use client";

import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";
import { Bar, Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, Legend, Tooltip);

export function LineChart({ data, options }) {
  return <Line data={data} options={options} />;
}

export function BarChart({ data, options }) {
  return <Bar data={data} options={options} />;
}
