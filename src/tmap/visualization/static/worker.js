/**
 * TMAP Binary Decoder Worker
 *
 * Decodes gzip-compressed binary data and transfers typed arrays
 * to the main thread using transferables (zero-copy).
 *
 * Uses fflate for decompression (loaded via importScripts).
 */

// fflate will be loaded dynamically
let fflate = null;

/**
 * Initialize fflate library
 */
async function initFflate(fflateUrl) {
  if (fflate) return;

  // Dynamic import for fflate
  const response = await fetch(fflateUrl);
  const code = await response.text();

  // Create a module from the code
  const blob = new Blob([code], { type: "application/javascript" });
  const url = URL.createObjectURL(blob);

  // For ES module fflate, we need to handle it differently
  // fflate UMD build exposes global `fflate`
  self.fflate = {};
  const script = code + "\nself.fflate = fflate;";
  eval(script);
  fflate = self.fflate;
}

/**
 * Decode gzip-compressed base64 data to typed array
 */
function decodeGzipBase64(base64Data, dtype) {
  // Decode base64
  const binaryString = atob(base64Data);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  // Decompress
  const decompressed = fflate.gunzipSync(bytes);

  // Create typed array based on dtype
  switch (dtype) {
    case "uint16":
      return new Uint16Array(decompressed.buffer);
    case "uint32":
      return new Uint32Array(decompressed.buffer);
    case "float32":
      return new Float32Array(decompressed.buffer);
    case "int32":
      return new Int32Array(decompressed.buffer);
    default:
      throw new Error(`Unknown dtype: ${dtype}`);
  }
}

/**
 * Dequantize Uint16 coordinates to Float32
 * Input: interleaved [x0, y0, x1, y1, ...] as Uint16
 * Output: { x: Float32Array, y: Float32Array }
 */
function dequantizeCoords(quantized, bits = 16) {
  const n = quantized.length / 2;
  const maxVal = bits === 16 ? 65535 : 4294967295;
  const scale = 2.0 / maxVal;

  // Create interleaved Float32Array for GPU efficiency
  const xy = new Float32Array(n * 2);

  for (let i = 0; i < n; i++) {
    xy[i * 2] = quantized[i * 2] * scale - 1.0; // x
    xy[i * 2 + 1] = quantized[i * 2 + 1] * scale - 1.0; // y
  }

  return xy;
}

/**
 * Decode coordinates from gzip-compressed base64
 */
function decodeCoords(base64Data, bits = 16) {
  const dtype = bits === 16 ? "uint16" : "uint32";
  const quantized = decodeGzipBase64(base64Data, dtype);
  return dequantizeCoords(quantized, bits);
}

/**
 * Decode a numeric column from gzip-compressed base64
 */
function decodeNumericColumn(base64Data, dtype) {
  return decodeGzipBase64(base64Data, dtype);
}

/**
 * Decode categorical column (dictionary indices)
 */
function decodeCategoricalColumn(base64Data, dictionary) {
  const indices = decodeGzipBase64(base64Data, "uint32");
  // Return indices and dictionary separately - main thread will map as needed
  return { indices, dictionary };
}

/**
 * Message handler
 */
self.onmessage = async function (e) {
  const { type, id, data } = e.data;

  try {
    switch (type) {
      case "init": {
        // Initialize fflate
        await initFflate(data.fflateUrl);
        self.postMessage({ type: "ready", id });
        break;
      }

      case "decode_coords": {
        const { base64, bits } = data;
        const xy = decodeCoords(base64, bits || 16);
        // Transfer the buffer (zero-copy)
        self.postMessage(
          { type: "coords", id, xy },
          [xy.buffer]
        );
        break;
      }

      case "decode_numeric": {
        const { base64, dtype, name } = data;
        const values = decodeNumericColumn(base64, dtype);
        self.postMessage(
          { type: "numeric", id, name, values },
          [values.buffer]
        );
        break;
      }

      case "decode_categorical": {
        const { base64, dictionary, name } = data;
        const result = decodeCategoricalColumn(base64, dictionary);
        self.postMessage(
          { type: "categorical", id, name, ...result },
          [result.indices.buffer]
        );
        break;
      }

      case "decode_all": {
        // Batch decode all sections
        const { header, coords, columns, fflateUrl } = data;

        // Ensure fflate is loaded
        await initFflate(fflateUrl);

        const result = {
          xy: null,
          columns: {},
        };
        const transferables = [];

        // Progress tracking
        const totalSections = 1 + Object.keys(columns).length;
        let completedSections = 0;

        const reportProgress = () => {
          completedSections++;
          self.postMessage({
            type: "progress",
            id,
            progress: completedSections / totalSections,
          });
        };

        // Decode coordinates
        const bits = header.coordBits || 16;
        result.xy = decodeCoords(coords, bits);
        transferables.push(result.xy.buffer);
        reportProgress();

        // Decode columns
        for (const [name, colData] of Object.entries(columns)) {
          const { base64, dtype, dictionary } = colData;

          if (dtype === "categorical" || dtype === "uint32") {
            // Categorical with dictionary
            if (dictionary) {
              const indices = decodeGzipBase64(base64, "uint32");
              result.columns[name] = { indices, dictionary };
              transferables.push(indices.buffer);
            } else {
              // Plain uint32
              const values = decodeGzipBase64(base64, "uint32");
              result.columns[name] = { values };
              transferables.push(values.buffer);
            }
          } else {
            // Numeric column
            const values = decodeNumericColumn(base64, dtype);
            result.columns[name] = { values };
            transferables.push(values.buffer);
          }
          reportProgress();
        }

        self.postMessage({ type: "complete", id, result }, transferables);
        break;
      }

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      id,
      error: error.message,
    });
  }
};
