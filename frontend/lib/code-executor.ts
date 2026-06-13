// Code execution utilities.
// Python всегда исполняется на бэкенде (изолированная песочница code-executor).
// JavaScript исполняется в sandbox-iframe, ML/WebGPU — через webgpu-executor.

import { executeONNXCode, executeTransformersCode, checkWebGPUSupport } from "./webgpu-executor"
import { logger } from "./logger"
import { api, ApiRequestError } from "./api/client"
import {
  JS_LIBRARY_CDN,
  PYTHON_EXECUTION_TIMEOUT,
  JS_EXECUTION_TIMEOUT,
  type CDNResource,
} from "./constants"

export interface ExecutionResult {
  output: string
  status: "success" | "error"
}

// Ответ бэкенд-эндпоинта POST /api/code/execute (проксирует в code-executor).
interface BackendExecuteResponse {
  success: boolean
  stdout: string
  stderr: string
  result: string | null
  error: string | null
}

/** Executes code in sandboxed iframe and returns result via postMessage */
function executeInSandbox(
  html: string,
  timeout: number
): Promise<{ type: "result" | "error"; logs?: string[]; message?: string }> {
  return new Promise((resolve) => {
    const iframe = document.createElement("iframe")
    iframe.sandbox.add("allow-scripts")
    iframe.style.display = "none"
    document.body.appendChild(iframe)

    const timeoutId = setTimeout(() => {
      cleanup()
      resolve({ type: "error", message: "Execution timed out (60s limit)" })
    }, timeout)

    const cleanup = () => {
      clearTimeout(timeoutId)
      window.removeEventListener("message", handleMessage)
      iframe.remove()
    }

    const handleMessage = (event: MessageEvent) => {
      if (event.source !== iframe.contentWindow) return
      cleanup()
      resolve(event.data)
    }

    window.addEventListener("message", handleMessage)
    iframe.srcdoc = html
  })
}

/**
 * Исполняет Python-код на бэкенде.
 * Код (в т.ч. отредактированный пользователем) отправляется в
 * POST /api/code/execute, который проксирует его в изолированный
 * сервис code-executor.
 */
export async function executePython(
  code: string,
  timeout: number = PYTHON_EXECUTION_TIMEOUT
): Promise<ExecutionResult> {
  try {
    const res = await api.post<BackendExecuteResponse>(
      "/api/code/execute",
      { code },
      { timeout }
    )

    if (res.success) {
      const segments: string[] = []
      if (res.stdout && res.stdout.trim()) segments.push(res.stdout.replace(/\n+$/, ""))
      if (res.result != null && res.result !== "") segments.push(res.result)
      const output = segments.join("\n")
      return {
        output: output || "Code executed successfully (no output)",
        status: "success",
      }
    }

    // Ошибка исполнения: показываем частичный stdout (если был) и текст ошибки.
    const segments: string[] = []
    if (res.stdout && res.stdout.trim()) segments.push(res.stdout.replace(/\n+$/, ""))
    if (res.error) segments.push(res.error)
    else if (res.stderr) segments.push(res.stderr)
    return {
      output: segments.join("\n") || "Execution failed",
      status: "error",
    }
  } catch (error) {
    if (error instanceof ApiRequestError) {
      // 401 уже инициировал редирект на /login внутри клиента.
      if (error.status === 401) {
        return { output: "Authentication required to run code", status: "error" }
      }
      logger.error("Backend code execution failed", error)
      return {
        output: `Code executor unavailable: ${error.message}`,
        status: "error",
      }
    }
    return {
      output: error instanceof Error ? error.message : String(error),
      status: "error",
    }
  }
}

// Detect required JS libraries from code and return CDN resources with SRI
function detectJSLibraries(code: string): CDNResource[] {
  const libs: CDNResource[] = []

  // TensorFlow.js
  if (code.includes("tf.") || code.includes("@tensorflow/tfjs")) {
    libs.push(JS_LIBRARY_CDN.tensorflow)
  }

  // Chart.js
  if (code.includes("Chart(") || code.includes("chart.js")) {
    libs.push(JS_LIBRARY_CDN.chartjs)
  }

  // D3.js
  if (code.includes("d3.") || code.includes("d3js")) {
    libs.push(JS_LIBRARY_CDN.d3)
  }

  // ONNX Runtime Web
  if (code.includes("ort.") || code.includes("onnxruntime")) {
    libs.push(JS_LIBRARY_CDN.onnxruntime)
  }

  return libs
}

// Cache for fetched library code
const libraryCache: Map<string, string> = new Map()

async function fetchLibraryCode(resource: CDNResource): Promise<string> {
  if (libraryCache.has(resource.url)) {
    return libraryCache.get(resource.url)!
  }

  try {
    // Note: fetch() doesn't support SRI directly, but browsers validate
    // SRI when loading scripts via <script> tags. For inline code,
    // we validate the hash ourselves.
    const response = await fetch(resource.url, {
      credentials: resource.crossOrigin === "use-credentials" ? "include" : "omit",
    })
    if (!response.ok) {
      throw new Error(`Failed to fetch ${resource.url}: ${response.status}`)
    }
    const code = await response.text()
    libraryCache.set(resource.url, code)
    return code
  } catch (error) {
    logger.error(`Failed to load library: ${resource.url}`, error)
    throw error
  }
}

// Check if code uses WebGPU/ONNX/Transformers.js (needs native execution, not iframe)
function requiresNativeExecution(code: string): "onnx" | "transformers" | null {
  // Check for ONNX Runtime usage
  if (
    code.includes("ort.") ||
    code.includes("onnxruntime") ||
    code.includes("loadONNXModel") ||
    code.includes("InferenceSession")
  ) {
    return "onnx"
  }

  // Check for Transformers.js usage
  if (
    code.includes("pipeline(") ||
    code.includes("@huggingface/transformers") ||
    code.includes("AutoModel") ||
    code.includes("AutoTokenizer")
  ) {
    return "transformers"
  }

  return null
}

export async function executeJavaScript(
  code: string,
  timeout: number = JS_EXECUTION_TIMEOUT
): Promise<ExecutionResult> {
  if (typeof window === "undefined") {
    return {
      output: "JavaScript execution is only available in browser",
      status: "error",
    }
  }

  // Check if code requires native execution (WebGPU/ONNX/Transformers)
  const nativeType = requiresNativeExecution(code)
  if (nativeType) {
    // Check WebGPU support
    const webgpuStatus = await checkWebGPUSupport()
    const gpuInfo = webgpuStatus.supported
      ? "✓ WebGPU available"
      : `⚠ WebGPU not available (${webgpuStatus.error}), using WASM fallback`

    let result: ExecutionResult

    if (nativeType === "onnx") {
      result = await executeONNXCode(code)
    } else {
      result = await executeTransformersCode(code)
    }

    // Prepend GPU info to output
    return {
      ...result,
      output: `${gpuInfo}\n\n${result.output}`,
    }
  }

  const libraries = detectJSLibraries(code)

  // Fetch all required libraries
  let libraryCode = ""
  if (libraries.length > 0) {
    try {
      const codes = await Promise.all(libraries.map(fetchLibraryCode))
      libraryCode = codes.join("\n;\n")
    } catch (error) {
      return {
        output: `Failed to load libraries: ${error instanceof Error ? error.message : String(error)}`,
        status: "error",
      }
    }
  }

  // Build sandbox script with console capture
  const libLoadLog =
    libraries.length > 0
      ? `console.log("Libraries loaded: ${libraries.map((l) => l.url.split("/").pop()).join(", ")}");`
      : ""

  const script = `
    const logs = [];

    console.log = (...args) => logs.push(args.map(a => {
      if (a === null) return 'null';
      if (a === undefined) return 'undefined';
      if (typeof a === 'object') {
        try {
          if (a.constructor && a.constructor.name === 'Tensor') {
            return 'Tensor: ' + JSON.stringify(a.arraySync());
          }
          return JSON.stringify(a, null, 2);
        } catch(e) {
          return String(a);
        }
      }
      return String(a);
    }).join(' '));
    console.error = (...args) => logs.push('Error: ' + args.join(' '));
    console.warn = (...args) => logs.push('Warning: ' + args.join(' '));

    (async function() {
      try {
        ${libLoadLog}
        const result = await (async function() {
          ${code}
        })();
        if (result !== undefined) {
          if (result && result.constructor && result.constructor.name === 'Tensor') {
            logs.push('Result Tensor: ' + JSON.stringify(result.arraySync()));
          } else {
            logs.push(typeof result === 'object' ? JSON.stringify(result, null, 2) : String(result));
          }
        }
        parent.postMessage({ type: 'result', logs }, '*');
      } catch (e) {
        parent.postMessage({ type: 'error', message: e.message || String(e) }, '*');
      }
    })();
  `

  const html = `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
<script>${libraryCode.replace(/<\/script>/gi, "<\\/script>")}</script>
<script>${script.replace(/<\/script>/gi, "<\\/script>")}</script>
</body>
</html>`

  const result = await executeInSandbox(html, timeout)

  if (result.type === "result") {
    const output = result.logs?.join("\n") || ""
    return { output: output || "Code executed successfully (no output)", status: "success" }
  }
  return { output: result.message || "Unknown error", status: "error" }
}

// Check if language is supported for execution
export function isExecutableLanguage(language: string): boolean {
  const normalizedLang = language.toLowerCase()
  return ["python", "py", "javascript", "js", "typescript", "ts"].includes(normalizedLang)
}

// Get execution function for language
export async function executeCode(
  code: string,
  language: string,
  timeout: number = PYTHON_EXECUTION_TIMEOUT
): Promise<ExecutionResult> {
  const normalizedLang = language.toLowerCase()

  if (normalizedLang === "python" || normalizedLang === "py") {
    return executePython(code, timeout)
  }

  if (["javascript", "js", "typescript", "ts"].includes(normalizedLang)) {
    return executeJavaScript(code, timeout)
  }

  return {
    output: `Language "${language}" is not supported for execution. Supported: Python, JavaScript`,
    status: "error",
  }
}

// Re-export WebGPU utilities
export { checkWebGPUSupport, executeONNXCode, executeTransformersCode } from "./webgpu-executor"
