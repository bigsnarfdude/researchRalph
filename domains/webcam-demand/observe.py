#!/usr/bin/env python3
"""
Webcam demand observer — grabs a frame from a traffic camera YouTube URL,
sends it to Gemini Flash Lite vision, extracts vehicle counts per approach.

Outputs (to stdout):
  ns_count: <int>       vehicles in north-south approaches
  ew_count: <int>       vehicles in east-west approaches
  total_count: <int>    all visible vehicles
  ns_ew_ratio: <float>  ns_count / max(ew_count, 1)
  confidence: <str>     high / medium / low
  raw_description: <one-line summary from model>
  success: True
"""
import sys
import os
import json
import subprocess
import tempfile
import yaml
from pathlib import Path

import google.genai as genai
from google.genai import types

# Override with WEBCAM_MODEL env var. gemini-3.1-flash-lite-preview when capacity allows.
MODEL = os.environ.get("WEBCAM_MODEL", "gemini-2.5-flash")


def grab_frame(url: str, output_path: str) -> str | None:
    """Extract a single live frame from a YouTube stream using yt-dlp + ffmpeg."""
    jpg_path = output_path + ".jpg"
    try:
        # Get direct stream URL from YouTube
        stream_result = subprocess.run(
            ["yt-dlp", url, "--skip-download", "--get-url", "-f", "best[height<=480]"],
            capture_output=True, text=True, timeout=30
        )
        stream_url = stream_result.stdout.strip().split("\n")[0]
        if not stream_url:
            raise ValueError("no stream URL returned")

        # Extract single frame with ffmpeg
        subprocess.run(
            ["ffmpeg", "-i", stream_url, "-vframes", "1", "-q:v", "2",
             "-update", "1", jpg_path, "-y"],
            capture_output=True, timeout=30, check=True
        )
        if Path(jpg_path).exists():
            return jpg_path
        return None
    except Exception as e:
        print(f"frame grab error: {e}", file=sys.stderr)
        return None


def count_vehicles(image_path: str, prompt_template: str, approaches: dict) -> dict:
    """Send image to Gemini Flash Lite and extract vehicle counts."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Detect mime type
    suffix = Path(image_path).suffix.lower()
    mime = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".webp": "image/webp",
            ".png": "image/png"}.get(suffix, "image/jpeg")

    prompt = prompt_template.format(
        ns_label=approaches.get("ns_label", "north-south"),
        ew_label=approaches.get("ew_label", "east-west"),
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
    )

    text = response.text.strip()

    # Try to parse JSON from response
    try:
        # Look for JSON block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return data
    except Exception:
        pass

    # Fallback: parse free text for numbers
    import re
    ns = 0
    ew = 0
    ns_match = re.search(r'(?:north.south|NS)[^\d]*(\d+)', text, re.IGNORECASE)
    ew_match = re.search(r'(?:east.west|EW)[^\d]*(\d+)', text, re.IGNORECASE)
    if ns_match:
        ns = int(ns_match.group(1))
    if ew_match:
        ew = int(ew_match.group(1))

    return {
        "ns_count": ns,
        "ew_count": ew,
        "confidence": "low",
        "raw_description": text[:120].replace("\n", " "),
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    camera_url = cfg.get("camera_url", "")
    prompt_template = cfg.get("prompt_template",
        "You are a traffic analyst. Count the vehicles waiting at each approach of this intersection.\n"
        "Return JSON only: {{\"ns_count\": <int>, \"ew_count\": <int>, "
        "\"confidence\": \"high|medium|low\", \"raw_description\": \"<one sentence>\"}}\n"
        "NS = {ns_label} direction. EW = {ew_label} direction."
    )
    approaches = {
        "ns_label": cfg.get("ns_label", "north-south"),
        "ew_label": cfg.get("ew_label", "east-west"),
    }

    # Grab frame
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_base = os.path.join(tmpdir, "frame")
        frame_path = grab_frame(camera_url, frame_base)

        if not frame_path:
            print("error: could not download frame", file=sys.stderr)
            print("success: False")
            sys.exit(1)

        # Analyze with Gemini
        result = count_vehicles(frame_path, prompt_template, approaches)

    ns = int(result.get("ns_count", 0))
    ew = int(result.get("ew_count", 0))
    total = ns + ew
    ratio = round(ns / max(ew, 1), 3)
    confidence = result.get("confidence", "unknown")
    description = result.get("raw_description", "")[:120].replace("\n", " ")

    print(f"ns_count: {ns}")
    print(f"ew_count: {ew}")
    print(f"total_count: {total}")
    print(f"ns_ew_ratio: {ratio}")
    print(f"confidence: {confidence}")
    print(f"raw_description: {description}")
    print("success: True")

    # Write demand.json for traffic-signal domain to consume
    demand = {
        "ns_rate": round(ns / 50.0, 3),   # normalize to per-step rate (50 steps ≈ 30s real)
        "ew_rate": round(ew / 50.0, 3),
        "ns_ew_ratio": ratio,
        "confidence": confidence,
        "source": camera_url,
    }
    demand_path = os.path.join(os.path.dirname(config_path), "demand.json")
    with open(demand_path, "w") as f:
        json.dump(demand, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        print("success: False")
        sys.exit(1)
