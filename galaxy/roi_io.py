# -*- coding: utf-8 -*-
"""ROI save/load helpers for GalaXY.

We store Napari Shapes vertices in Napari-native order (Y,X) to avoid ambiguity.

JSON schema:
{
  "app": "GalaXY",
  "format_version": 1,
  "created_utc": "2026-02-11T09:12:33Z",
  "rois": {
    "domains": [{"name":"...", "vertices_yx":[[y,x], ...]}],
    "holes":   [{"name":"...", "type":"nucleus", "vertices_yx":[[y,x], ...]}],
    "seeds":   [{"name":"...", "vertices_yx":[[y,x], ...]}]
  }
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


FORMAT_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def save_rois_json(path: str | Path, rois: Dict[str, List[dict]], *, app: str = "GalaXY") -> None:
    path = Path(path)
    payload = {
        "app": app,
        "format_version": FORMAT_VERSION,
        "created_utc": _utc_now_iso(),
        "rois": {
            "domains": rois.get("domains", []),
            "holes": rois.get("holes", []),
            "seeds": rois.get("seeds", []),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_rois_json(path: str | Path) -> Dict[str, List[dict]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rois = payload.get("rois", payload)  # tolerate old/flat schema

    def _norm_list(key: str) -> List[dict]:
        lst = rois.get(key, []) or []
        out: List[dict] = []
        for i, it in enumerate(lst):
            if not isinstance(it, dict):
                continue
            name = str(it.get("name") or f"{key[:-1]}_{i+1}")
            verts = it.get("vertices_yx") or it.get("vertices") or it.get("verts_yx")
            if verts is None:
                continue
            out_it = {"name": name, "vertices_yx": verts}
            if key == "holes":
                out_it["type"] = str(it.get("type") or "nucleus")
            out.append(out_it)
        return out

    return {
        "domains": _norm_list("domains"),
        "holes": _norm_list("holes"),
        "seeds": _norm_list("seeds"),
    }
