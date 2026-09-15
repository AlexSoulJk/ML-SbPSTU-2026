export async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const init = { ...options, headers };

  if (init.body && !(init.body instanceof FormData) && typeof init.body !== "string") {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(init.body);
  } else if (typeof init.body === "string") {
    headers["Content-Type"] = headers["Content-Type"] || "application/json";
  }

  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = await response.text();
    try {
      detail = JSON.parse(detail).detail || detail;
    } catch (_error) {
      // Keep raw detail.
    }
    throw new Error(detail);
  }

  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response.text();
}
