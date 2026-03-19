"use client";

import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";
import { Bar, Line, Pie } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, ArcElement, BarElement, LineElement, PointElement, Legend, Tooltip);

export function LineChart({ data, options }) {
  return <Line data={data} options={options} />;
}

export function BarChart({ data, options }) {
  return <Bar data={data} options={options} />;
}

export function PieChart({ data, options }) {
  return <Pie data={data} options={options} />;
}
