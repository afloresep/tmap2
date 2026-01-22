#!/usr/bin/env python3
"""
Export TMAP visualization as a standalone HTML file.

The generated HTML file contains all data embedded as base64,
so it can be opened directly in a browser without a server.

Usage:
    # From .npy coordinates
    python export_standalone.py --npy x_y_coords.npy -o my_tmap.html

    # From layout.bin (already generated)
    python export_standalone.py --bin layout.bin -o my_tmap.html

    # With title and metadata
    python export_standalone.py --npy coords.npy -o output.html --title "My TMAP"
"""

import argparse
import base64
import struct
from pathlib import Path

import numpy as np


def normalize_coordinates(x: np.ndarray, y: np.ndarray, padding: float = 0.05) -> tuple:
    """Normalize to [-1, 1] preserving aspect ratio."""
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0
    max_range = max(x_range, y_range)

    scale = 2.0 * (1.0 - padding) / max_range

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    x_norm = ((x - x_center) * scale).astype(np.float32)
    y_norm = ((y - y_center) * scale).astype(np.float32)

    return x_norm, y_norm


def load_npy_coords(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load coordinates from .npy file (shape 2×N or N×2)."""
    data = np.load(path)
    if data.shape[0] == 2:
        return data[0].astype(np.float32), data[1].astype(np.float32)
    elif data.shape[1] == 2:
        return data[:, 0].astype(np.float32), data[:, 1].astype(np.float32)
    else:
        raise ValueError(f"Cannot interpret shape {data.shape} as coordinates")


def create_binary_data(
    x: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray | None = None,
) -> bytes:
    """Create binary layout data."""
    n_points = len(x)
    x_norm, y_norm = normalize_coordinates(x, y)

    parts = [
        struct.pack("<I", n_points),
        x_norm.tobytes(),
        y_norm.tobytes(),
    ]

    if categories is not None:
        parts.append(categories.astype(np.float32).tobytes())

    return b"".join(parts)


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }}
        #app {{ display: flex; flex-direction: column; height: 100vh; }}
        header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 20px;
            background: #16213e;
            border-bottom: 1px solid #0f3460;
        }}
        header h1 {{ font-size: 1.2rem; font-weight: 500; }}
        #stats {{ display: flex; gap: 20px; font-size: 0.85rem; color: #aaa; }}
        .stat-value {{ color: #4fc3f7; font-weight: 600; margin-left: 6px; }}
        #canvas-container {{ flex: 1; position: relative; }}
        #scatter-canvas {{ width: 100%; height: 100%; display: block; }}

        #status {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            background: rgba(22, 33, 62, 0.95);
            padding: 30px 50px;
            border-radius: 12px;
        }}
        #status.hidden {{ display: none; }}
        #status-spinner {{
            width: 40px; height: 40px;
            border: 3px solid #0f3460;
            border-top-color: #4fc3f7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 16px;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

        #controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        #controls.hidden {{ display: none; }}
        .control-group {{
            background: rgba(22, 33, 62, 0.9);
            border-radius: 6px;
            padding: 10px;
        }}
        .control-group label {{ display: block; font-size: 0.75rem; color: #aaa; margin-bottom: 4px; }}
        .control-group input[type="range"] {{ width: 120px; }}
        button {{
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: #0f3460;
            color: #eee;
            cursor: pointer;
        }}
        button:hover {{ background: #1a5276; }}

        #perf {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(22, 33, 62, 0.9);
            border-radius: 6px;
            padding: 10px;
            font-size: 0.75rem;
            font-family: monospace;
        }}
        #perf.hidden {{ display: none; }}
    </style>
</head>
<body>
    <div id="app">
        <header>
            <h1>{title}</h1>
            <div id="stats">
                <span>Points:<span class="stat-value" id="stat-points">{n_points}</span></span>
                <span>Draw:<span class="stat-value" id="stat-draw">-</span></span>
                <span>FPS:<span class="stat-value" id="stat-fps">-</span></span>
            </div>
        </header>

        <div id="canvas-container">
            <canvas id="scatter-canvas"></canvas>

            <div id="status">
                <div id="status-spinner"></div>
                <p id="status-text">Loading...</p>
            </div>

            <div id="controls" class="hidden">
                <div class="control-group">
                    <label>Point Size: <span id="size-val">1</span></label>
                    <input type="range" id="point-size" min="0.5" max="10" step="0.5" value="1">
                </div>
                <div class="control-group">
                    <label>Opacity: <span id="opacity-val">0.5</span></label>
                    <input type="range" id="opacity" min="0.1" max="1" step="0.1" value="0.5">
                </div>
                <div class="control-group">
                    <button id="reset-btn">Reset View</button>
                </div>
            </div>

            <div id="perf" class="hidden"></div>
        </div>
    </div>

    <!-- Embedded binary data as base64 -->
    <script id="layout-data" type="application/octet-stream">{data_base64}</script>

    <script type="module">
        import createScatterplot from 'https://cdn.jsdelivr.net/npm/regl-scatterplot@1.10.1/+esm';

        var $ = function(id) {{ return document.getElementById(id); }};
        var canvas = $('scatter-canvas');
        var container = $('canvas-container');
        var scatterplot = null;

        function setStatus(text) {{
            $('status').classList.remove('hidden');
            $('status-text').textContent = text;
        }}

        function hideStatus() {{
            $('status').classList.add('hidden');
            $('controls').classList.remove('hidden');
            $('perf').classList.remove('hidden');
        }}

        function yieldToUI() {{
            return new Promise(function(resolve) {{ setTimeout(resolve, 10); }});
        }}

        function resizeCanvas() {{
            var rect = container.getBoundingClientRect();
            var dpr = window.devicePixelRatio || 1;
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            if (scatterplot) {{
                scatterplot.set({{ width: rect.width, height: rect.height }});
            }}
        }}
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        function decodeBase64ToArrayBuffer(base64) {{
            var binaryString = atob(base64);
            var len = binaryString.length;
            var bytes = new Uint8Array(len);
            for (var i = 0; i < len; i++) {{
                bytes[i] = binaryString.charCodeAt(i);
            }}
            return bytes.buffer;
        }}

        async function main() {{
            try {{
                setStatus('Decoding data...');
                await yieldToUI();

                // Get embedded base64 data
                var base64Data = $('layout-data').textContent.trim();
                var buffer = decodeBase64ToArrayBuffer(base64Data);

                setStatus('Parsing points...');
                await yieldToUI();

                var dataView = new DataView(buffer);
                var numPoints = dataView.getUint32(0, true);

                // Copy data to new arrays
                var x = new Float32Array(numPoints);
                var y = new Float32Array(numPoints);
                var srcX = new Float32Array(buffer, 4, numPoints);
                var srcY = new Float32Array(buffer, 4 + numPoints * 4, numPoints);
                x.set(srcX);
                y.set(srcY);

                // Categories if present
                var categories = null;
                var expectedSize = 4 + numPoints * 8;
                if (buffer.byteLength > expectedSize) {{
                    categories = new Float32Array(numPoints);
                    var srcCat = new Float32Array(buffer, 4 + numPoints * 8, numPoints);
                    categories.set(srcCat);
                }}

                setStatus('Initializing WebGL...');
                await yieldToUI();

                var rect = container.getBoundingClientRect();
                var isLarge = numPoints > 500000;
                var isVeryLarge = numPoints >= 1000000;

                scatterplot = createScatterplot({{
                    canvas: canvas,
                    width: rect.width,
                    height: rect.height,
                    pointSize: isVeryLarge ? 1 : (isLarge ? 1.5 : 3),
                    opacity: isVeryLarge ? 0.5 : (isLarge ? 0.6 : 0.8),
                    pointColor: ['#4fc3f7', '#ff7043', '#66bb6a', '#ab47bc', '#ffd54f'],
                    colorBy: 'valueA',
                    performanceMode: isLarge,
                    spatialIndexUseWorker: false,
                    showReticle: !isVeryLarge,
                    reticleColor: [1, 1, 0.878, 0.5]
                }});

                setStatus('Drawing ' + numPoints.toLocaleString() + ' points...');
                await yieldToUI();

                var points = {{ x: x, y: y }};
                if (categories) {{
                    points.z = categories;
                }} else {{
                    var cats = new Float32Array(numPoints);
                    for (var k = 0; k < numPoints; k++) {{ cats[k] = k % 5; }}
                    points.z = cats;
                }}

                var drawStart = performance.now();
                await scatterplot.draw(points);
                var drawTime = performance.now() - drawStart;

                $('stat-draw').textContent = drawTime.toFixed(0) + 'ms';
                hideStatus();

                // Controls
                $('point-size').oninput = function(e) {{
                    $('size-val').textContent = e.target.value;
                    scatterplot.set({{ pointSize: parseFloat(e.target.value) }});
                }};
                $('opacity').oninput = function(e) {{
                    $('opacity-val').textContent = e.target.value;
                    scatterplot.set({{ opacity: parseFloat(e.target.value) }});
                }};
                $('reset-btn').onclick = function() {{ scatterplot.reset(); }};

                // FPS counter
                var frames = 0, lastTime = performance.now();
                function updateFPS() {{
                    frames++;
                    var now = performance.now();
                    if (now - lastTime >= 1000) {{
                        var fps = Math.round(frames * 1000 / (now - lastTime));
                        $('stat-fps').textContent = fps;
                        $('perf').textContent = 'FPS: ' + fps + ' | Points: ' + numPoints.toLocaleString();
                        frames = 0;
                        lastTime = now;
                    }}
                    requestAnimationFrame(updateFPS);
                }}
                requestAnimationFrame(updateFPS);

                if (!isVeryLarge) {{
                    scatterplot.subscribe('pointOver', function(idx) {{
                        $('perf').textContent = 'Point #' + idx.toLocaleString();
                    }});
                    scatterplot.subscribe('pointOut', function() {{
                        $('perf').textContent = 'Points: ' + numPoints.toLocaleString();
                    }});
                }}

            }} catch (err) {{
                console.error(err);
                $('status-text').textContent = 'Error: ' + err.message;
            }}
        }}

        main();
    </script>
</body>
</html>
'''


def export_standalone(
    x: np.ndarray,
    y: np.ndarray,
    output_path: str,
    title: str = "TMAP Visualization",
    categories: np.ndarray | None = None,
) -> dict:
    """
    Export visualization as standalone HTML file.

    Args:
        x: X coordinates
        y: Y coordinates
        output_path: Output HTML file path
        title: Page title
        categories: Optional category indices for coloring

    Returns:
        Dict with file path and stats
    """
    n_points = len(x)
    print(f"Exporting {n_points:,} points to standalone HTML...")

    # Create binary data
    binary_data = create_binary_data(x, y, categories)
    binary_size_mb = len(binary_data) / 1024 / 1024

    # Encode as base64
    data_base64 = base64.b64encode(binary_data).decode("ascii")
    base64_size_mb = len(data_base64) / 1024 / 1024

    print(f"  Binary size: {binary_size_mb:.1f} MB")
    print(f"  Base64 size: {base64_size_mb:.1f} MB")

    # Generate HTML
    html = HTML_TEMPLATE.format(
        title=title,
        n_points=f"{n_points:,}",
        data_base64=data_base64,
    )

    # Write file
    output_path = Path(output_path)
    output_path.write_text(html)

    html_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  HTML size: {html_size_mb:.1f} MB")
    print(f"  Output: {output_path}")

    return {
        "path": str(output_path),
        "n_points": n_points,
        "binary_mb": binary_size_mb,
        "html_mb": html_size_mb,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export TMAP as standalone HTML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--npy", help="Path to .npy coordinates (shape 2×N or N×2)")
    parser.add_argument("--bin", help="Path to existing layout.bin file")
    parser.add_argument("-o", "--output", default="tmap_standalone.html", help="Output HTML file")
    parser.add_argument("--title", default="TMAP Visualization", help="Page title")
    parser.add_argument("--synthetic", type=int, help="Generate N synthetic points for testing")

    args = parser.parse_args()

    if args.npy:
        x, y = load_npy_coords(args.npy)
        categories = None
    elif args.bin:
        # Read existing binary file
        with open(args.bin, "rb") as f:
            buffer = f.read()
        n_points = struct.unpack("<I", buffer[:4])[0]
        x = np.frombuffer(buffer[4 : 4 + n_points * 4], dtype=np.float32)
        y = np.frombuffer(buffer[4 + n_points * 4 : 4 + n_points * 8], dtype=np.float32)
        # Already normalized, but export_standalone will re-normalize (harmless)
        categories = None
        if len(buffer) > 4 + n_points * 8:
            categories = np.frombuffer(buffer[4 + n_points * 8 :], dtype=np.float32)
    elif args.synthetic:
        print(f"Generating {args.synthetic:,} synthetic points...")
        np.random.seed(42)
        n = args.synthetic
        x = (np.random.randn(n) * 500).astype(np.float32)
        y = (np.random.randn(n) * 500).astype(np.float32)
        categories = (np.arange(n) % 5).astype(np.float32)
    else:
        parser.error("Specify --npy, --bin, or --synthetic")

    export_standalone(x, y, args.output, args.title, categories)


if __name__ == "__main__":
    main()
