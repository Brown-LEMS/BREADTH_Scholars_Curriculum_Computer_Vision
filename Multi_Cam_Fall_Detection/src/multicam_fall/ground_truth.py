from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


POSITION_CODE_MAP = {
    1: {"label": "walking_standing_up", "description": "Walking, standing up"},
    2: {"label": "falling", "description": "Falling"},
    3: {"label": "lying_on_ground", "description": "Lying on the ground"},
    4: {"label": "crouching", "description": "Crouching"},
    5: {"label": "moving_down", "description": "Moving down"},
    6: {"label": "moving_up", "description": "Moving up"},
    7: {"label": "sitting", "description": "Sitting"},
    8: {"label": "lying_on_sofa", "description": "Lying on a sofa"},
    9: {"label": "moving_horizontally", "description": "Moving horizontaly"},
}


def parse_ground_truth_from_report(pdf_path: str | Path) -> Dict[str, object]:
    reader = PdfReader(str(pdf_path))
    five_number_pattern = re.compile(r"^(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$")
    three_number_pattern = re.compile(r"^(\d+)\s+(\d+)\s+(\d+)$")

    scenarios: Dict[str, Dict[str, object]] = {}
    current_scenario = None
    current_reference = None
    current_segments: List[Dict[str, object]] = []

    def flush_current() -> None:
        nonlocal current_scenario, current_reference, current_segments
        if current_scenario is None:
            return
        chute_name = f"chute{current_scenario:02d}"
        scenarios[chute_name] = {
            "report_scenario_number": current_scenario,
            "camera_reference": current_reference,
            "segments": current_segments,
        }
        current_scenario = None
        current_reference = None
        current_segments = []

    for page_index in range(16, 22):
        text = reader.pages[page_index].extract_text() or ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("scenario") or line.startswith("position noted are"):
                continue
            if line.startswith("This is done for all frames"):
                continue
            if line.isdigit() and len(line) <= 2:
                continue

            five_match = five_number_pattern.match(line)
            if five_match:
                flush_current()
                current_scenario = int(five_match.group(1))
                current_reference = int(five_match.group(2))
                start_frame = int(five_match.group(3))
                end_frame = int(five_match.group(4))
                code = int(five_match.group(5))
                current_segments.append(
                    {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "position_code": code,
                        "label": POSITION_CODE_MAP[code]["label"],
                        "description": POSITION_CODE_MAP[code]["description"],
                    }
                )
                continue

            three_match = three_number_pattern.match(line)
            if three_match and current_scenario is not None:
                start_frame = int(three_match.group(1))
                end_frame = int(three_match.group(2))
                code = int(three_match.group(3))
                current_segments.append(
                    {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "position_code": code,
                        "label": POSITION_CODE_MAP[code]["label"],
                        "description": POSITION_CODE_MAP[code]["description"],
                    }
                )

    flush_current()

    return {
        "source_pdf": str(Path(pdf_path).resolve()),
        "position_codes": {str(key): value for key, value in POSITION_CODE_MAP.items()},
        "scenarios": scenarios,
    }


def save_ground_truth_json(pdf_path: str | Path, output_path: str | Path) -> Path:
    parsed = parse_ground_truth_from_report(pdf_path)
    output = Path(output_path)
    output.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    return output


def load_ground_truth_json(path: str | Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label_for_frame(
    ground_truth: Dict[str, object],
    chute_name: str,
    frame_index: int,
) -> Dict[str, object] | None:
    scenario = ground_truth.get("scenarios", {}).get(chute_name)
    if not scenario:
        return None
    for segment in scenario.get("segments", []):
        if segment["start_frame"] <= frame_index <= segment["end_frame"]:
            return segment
    return None
