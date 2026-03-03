"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchApi } from "@/lib/api";

export default function ModelsPage() {
  const [models, setModels] = useState({});

  useEffect(() => {
    fetchApi("/api/models").then(setModels).catch(console.error);
  }, []);

  const modelNames = Object.keys(models);

  return (
    <>
      <h2 className="pageTitle">Model Pages</h2>
      <p className="subtle">Open a dynamic model route: /model/[modelId]</p>
      <div className="card">
        <div className="pillList">
          {modelNames.map((name) => (
            <Link className="pill" key={name} href={`/model/${encodeURIComponent(name)}`}>
              {name}
            </Link>
          ))}
        </div>
      </div>
    </>
  );
}
