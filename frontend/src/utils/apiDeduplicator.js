/**
 * Lightweight in-flight request deduplicator and short-lived cache.
 * Prevents concurrent React components from flooding the backend with duplicate GET calls.
 */

const inflightRequests = new Map();
const responseCache = new Map();

/**
 * Deduplicated fetch wrapper for JSON GET requests.
 *
 * @param {string} url - Request URL
 * @param {RequestInit} [options] - Standard fetch options
 * @param {number} [ttlMs=15000] - In-memory cache TTL in milliseconds (default: 15s)
 * @returns {Promise<any>} Parsed JSON response
 */
export async function deduplicatedFetch(url, options = {}, ttlMs = 15000) {
  const method = (options.method || 'GET').toUpperCase();

  // Only deduplicate/cache idempotent GET requests
  if (method !== 'GET') {
    // Invalidate any cached GET response for this endpoint
    responseCache.delete(url);
    const res = await fetch(url, options);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    return res.json();
  }

  const now = Date.now();

  // 1. Check short-lived cache
  if (responseCache.has(url)) {
    const { timestamp, data } = responseCache.get(url);
    if (now - timestamp < ttlMs) {
      return data;
    }
    responseCache.delete(url);
  }

  // 2. Check existing in-flight request
  if (inflightRequests.has(url)) {
    return inflightRequests.get(url);
  }

  // 3. Initiate new fetch and register in-flight promise
  const promise = (async () => {
    try {
      const res = await fetch(url, options);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      const data = await res.json();
      responseCache.set(url, { timestamp: Date.now(), data });
      return data;
    } finally {
      inflightRequests.delete(url);
    }
  })();

  inflightRequests.set(url, promise);
  return promise;
}

export function clearApiCache(url) {
  if (url) {
    responseCache.delete(url);
    inflightRequests.delete(url);
  } else {
    responseCache.clear();
    inflightRequests.clear();
  }
}
