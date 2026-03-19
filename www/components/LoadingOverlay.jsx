"use client";

import { LoaderCircle } from "lucide-react";

export default function LoadingOverlay({ text = "Loading data..." }) {
  return (
    <div className="pageLoadingOverlay" role="status" aria-live="polite" aria-label={text}>
      <div className="pageLoadingCard">
        <LoaderCircle size={34} className="pageLoadingIcon" />
        <div className="pageLoadingText">{text}</div>
      </div>
    </div>
  );
}
