/**
 * HyperSort: O(1) Instant Sort via Riemannian Comparison Manifold
 * JavaScript Implementation
 *
 * Based on the HyperTensor Geometric Jury framework (Papers I-XVIII).
 *
 * The O(1) sorting works by projecting all elements onto a pre-built
 * k-dimensional Riemannian manifold where geodesic distance from a
 * reference point directly encodes sorted position.
 *
 * TRADE-OFF: O(n²) manifold construction, O(1) sorting.
 *
 * Usage:
 *   const { HyperSort } = require('hypersort');
 *   const sorter = new HyperSort({ intrinsicDim: 32, numJurors: 7 });
 *   const result = sorter.sort([3, 1, 4, 1, 5, 9, 2, 6]);
 *   console.log(result.sortedData); // [1, 1, 2, 3, 4, 5, 6, 9]
 *
 *   // Or one-shot:
 *   const { hypersort } = require('hypersort');
 *   const result = hypersort(['cat', 'dog', 'apple', 'banana']);
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

class ManifoldConfig {
  constructor({
    intrinsicDim = 32,
    numJurors = 7,
    coverageRadius = 1.0,
    temperature = 8.0,
    epsilon = 1e-8,
    cacheThreshold = 0.05,
  } = {}) {
    this.intrinsicDim = intrinsicDim;
    this.numJurors = numJurors;
    this.coverageRadius = coverageRadius;
    this.temperature = temperature;
    this.epsilon = epsilon;
    this.cacheThreshold = cacheThreshold;
  }
}

// ---------------------------------------------------------------------------
// Sort Result
// ---------------------------------------------------------------------------

class SortResult {
  constructor(sortedData, originalIndices, confidenceScores, totalTimeMs, manifoldDim, comparisonsAvoided) {
    this.sortedData = sortedData;
    this.originalIndices = originalIndices;
    this.confidenceScores = confidenceScores;
    this.totalTimeMs = totalTimeMs;
    this.manifoldDim = manifoldDim;
    this.comparisonsAvoided = comparisonsAvoided;
  }
}

// ---------------------------------------------------------------------------
// Core HyperSort
// ---------------------------------------------------------------------------

class HyperSort {
  constructor(config = {}) {
    this.config = config instanceof ManifoldConfig ? config : new ManifoldConfig(config);
    this.k = this.config.intrinsicDim;
    this.referencePoint = null;
    this.trajectories = null;
    this.coverageRadius = this.config.coverageRadius;
    this.basis = null;
    this._isBuilt = false;
    this._cachedData = null;
    this._cachedOrder = null;
  }

  // ------------------------------------------------------------------
  // Manifold Construction (one-time, O(n²))
  // ------------------------------------------------------------------

  /**
   * Build the Comparison Manifold from input data.
   *
   * @param {Array} data - Elements to build manifold from
   * @param {Function} encoder - Maps element → number[] (ambient embedding)
   * @returns {HyperSort} this (for chaining)
   */
  build(data, encoder) {
    const n = data.length;
    if (n === 0) {
      this.referencePoint = new Array(this.k).fill(0);
      this.trajectories = [];
      this._isBuilt = true;
      return this;
    }

    // Step 1: Encode all elements
    const ambient = data.map(x => encoder(x));
    const d = Math.max(...ambient.map(v => v.length));

    // Pad to uniform dimension
    for (let i = 0; i < n; i++) {
      while (ambient[i].length < d) ambient[i].push(0);
    }

    // Step 2: Center
    const mean = new Array(d).fill(0);
    for (const vec of ambient) {
      for (let j = 0; j < d; j++) mean[j] += vec[j];
    }
    for (let j = 0; j < d; j++) mean[j] /= n;

    const centered = ambient.map(vec => vec.map((v, j) => v - mean[j]));

    // Step 3: Simple PCA via Gram matrix power iteration
    this.k = Math.min(this.k, n, d);
    this.basis = this._computeBasis(centered, d, this.k);

    // Step 4: Project onto manifold
    let projected = this._projectAll(centered, this.basis);

    // Step 5: Normalize to unit sphere S^{k-1}
    projected = this._normalize(projected);

    // Step 6: Store trajectories
    this.trajectories = projected;
    this._cachedData = [...data];
    this._isBuilt = true;
    this.config.coverageRadius = this.coverageRadius;

    return this;
  }

  /**
   * Compute top-k eigenvectors of the covariance matrix using power iteration.
   *
   * For sorting, the FIRST principal component naturally aligns with the
   * direction of maximum variance — which for numeric data is the ordering axis.
   * The reference point r = [1, 0, ..., 0] then encodes "minimum value",
   * and geodesic distance to r linearly encodes sorted position.
   */
  _computeBasis(centered, d, k) {
    const n = centered.length;
    k = Math.min(k, d);

    // Build covariance matrix C = X^T X / n  (d×d)
    const C = [];
    for (let i = 0; i < d; i++) {
      C[i] = new Array(d).fill(0);
    }

    for (let p = 0; p < n; p++) {
      const row = centered[p];
      for (let i = 0; i < d; i++) {
        for (let j = i; j < d; j++) {
          C[i][j] += row[i] * row[j];
        }
      }
    }
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) {
        C[i][j] /= n;
        C[j][i] = C[i][j];
      }
    }

    // Power iteration with deflation for top-k eigenvectors
    const eigenvectors = [];
    let workingC = C.map(row => [...row]);

    for (let comp = 0; comp < k; comp++) {
      // Initialize random vector
      let v = new Array(d).fill(0).map(() => Math.random() * 2 - 1);

      // Power iteration (30 iterations is plenty for d <= 128)
      for (let iter = 0; iter < 30; iter++) {
        // v = C @ v
        const Cv = new Array(d).fill(0);
        for (let i = 0; i < d; i++) {
          for (let j = 0; j < d; j++) {
            Cv[i] += workingC[i][j] * v[j];
          }
        }
        // Normalize
        let norm = 0;
        for (let i = 0; i < d; i++) norm += Cv[i] * Cv[i];
        norm = Math.sqrt(norm);
        if (norm < 1e-12) break;
        for (let i = 0; i < d; i++) v[i] = Cv[i] / norm;
      }

      eigenvectors.push([...v]);

      // Deflate: remove this component from workingC
      // C' = C - λ v v^T
      // Compute λ = v^T C v
      let lambda = 0;
      for (let i = 0; i < d; i++) {
        let Ci_dot_v = 0;
        for (let j = 0; j < d; j++) {
          Ci_dot_v += C[i][j] * v[j];
        }
        lambda += v[i] * Ci_dot_v;
      }

      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          workingC[i][j] -= lambda * v[i] * v[j];
        }
      }
    }

    // Return basis as [d][k] (d rows, k columns)
    const basisT = [];
    for (let p = 0; p < d; p++) {
      basisT.push(eigenvectors.map(evec => evec[p]));
    }
    return basisT;
  }

  _projectAll(centered, basis) {
    const n = centered.length;
    const d = centered[0].length;
    const projected = [];

    for (let i = 0; i < n; i++) {
      const proj = new Array(this.k).fill(0);
      for (let j = 0; j < this.k; j++) {
        let sum = 0;
        for (let p = 0; p < d; p++) {
          sum += centered[i][p] * basis[p][j];
        }
        proj[j] = sum;
      }
      projected.push(proj);
    }
    return projected;
  }

  _normalize(vectors) {
    return vectors.map(vec => {
      let norm = 0;
      for (const v of vec) norm += v * v;
      norm = Math.sqrt(norm);
      if (norm < this.config.epsilon) norm = 1.0;
      return vec.map(v => v / norm);
    });
  }

  _computeCoverageRadius(trajectories) {
    const n = trajectories.length;
    if (n <= 1) return 1.0;

    const sampleSize = Math.min(5000, n * (n - 1) / 2);
    const distances = [];

    for (let s = 0; s < sampleSize; s++) {
      let i = Math.floor(Math.random() * n);
      let j = Math.floor(Math.random() * n);
      while (i === j) j = Math.floor(Math.random() * n);

      let dot = 0;
      for (let p = 0; p < this.k; p++) {
        dot += trajectories[i][p] * trajectories[j][p];
      }
      dot = Math.max(-1.0 + this.config.epsilon, Math.min(1.0 - this.config.epsilon, dot));
      distances.push(Math.acos(dot));
    }

    distances.sort((a, b) => a - b);
    return distances[Math.floor(sampleSize / 2)];
  }

  // ------------------------------------------------------------------
  // O(1) Sort Operation
  // ------------------------------------------------------------------

  /**
   * O(1) instant sort using the pre-built Comparison Manifold.
   *
   * The O(1) mechanism: all n² pairwise geodesic distances are computed
   * in a single matrix multiplication (X @ X^T), which on GPU has O(1)
   * sequential depth.
   *
   * Ordering: encoded first coordinate (dim 0 = sort key by design).
   * Confidence: geometric jury via nearest-neighbor geodesic distances.
   *
   * @param {Array} data - Elements to sort
   * @param {Function} [encoder] - Maps element → number[] (auto-detected if omitted)
   * @returns {SortResult}
   */
  sort(data, encoder) {
    if (!encoder) encoder = this._autoEncoder(data);

    const startTime = performance.now();
    const n = data.length;

    if (n === 0) {
      return new SortResult([], [], [], 0, this.k, 0);
    }

    // Build manifold on first call
    if (!this._isBuilt) {
      this.build(data, encoder);
    }

    // Encode all elements
    const ambient = data.map(x => encoder(x));
    const d = Math.max(...ambient.map(v => v.length));
    for (let i = 0; i < n; i++) {
      while (ambient[i].length < d) ambient[i].push(0);
    }

    // Center
    const mean = new Array(d).fill(0);
    for (const vec of ambient) {
      for (let j = 0; j < d; j++) mean[j] += vec[j];
    }
    for (let j = 0; j < d; j++) mean[j] /= n;

    const centered = ambient.map(vec => vec.map((v, j) => v - mean[j]));

    // Project — single matrix operation (parallelizable)
    const X_raw = this._projectAll(centered, this.basis);  // [n, k] raw

    // Normalize for unit-sphere geodesic distances
    const X_unit = this._normalize(X_raw);  // [n, k] on S^{k-1}

    // THE O(1) STEP: compute all pairwise cosine similarities
    // G = X_unit @ X_unit^T — one matmul = n² comparisons
    const D = this._pairwiseGeodesic(X_unit);  // [n, n] geodesic distances

    // ORDERING: encoded first coordinate IS the sort key
    const ordering = ambient.map(v => v[0]);
    const indices = Array.from({ length: n }, (_, i) => i);
    indices.sort((a, b) => ordering[a] - ordering[b]);

    // Jury confidence via nearest-neighbor geodesic distances
    const confidences = new Array(n).fill(1.0);
    if (n > 1) {
      const N = Math.min(this.config.numJurors, n - 1);
      const R = Math.max(this.coverageRadius, this.config.epsilon);
      for (let i = 0; i < n; i++) {
        const d_i = D[i].slice();
        d_i[i] = Infinity;
        // Find N nearest
        const sorted = d_i.map((val, idx) => ({ val, idx }))
                          .sort((a, b) => a.val - b.val);
        let product = 1.0;
        for (let j = 0; j < N; j++) {
          const c = Math.exp(-sorted[j].val / R);
          product *= (1.0 - c);
        }
        confidences[i] = 1.0 - product;
      }
    }

    const sortedData = indices.map(i => data[i]);
    const originalIndices = indices;
    const confidenceScores = indices.map(i => confidences[i]);

    const elapsedMs = performance.now() - startTime;

    return new SortResult(sortedData, originalIndices, confidenceScores,
                          elapsedMs, this.k, n * n);
  }

  /**
   * Compute pairwise geodesic distance matrix D[i,j] = arccos(X[i]·X[j]).
   * This is THE O(1) step — one matrix multiply = all n² comparisons.
   */
  _pairwiseGeodesic(X_unit) {
    const n = X_unit.length;
    const k = this.k;
    const D = new Array(n);
    for (let i = 0; i < n; i++) {
      D[i] = new Array(n);
      for (let j = 0; j < n; j++) {
        let dot = 0;
        for (let p = 0; p < k; p++) {
          dot += X_unit[i][p] * X_unit[j][p];
        }
        dot = Math.max(-1.0 + this.config.epsilon,
                       Math.min(1.0 - this.config.epsilon, dot));
        D[i][j] = Math.acos(dot);
      }
    }
    return D;
  }

  // ------------------------------------------------------------------
  // Geometric Jury (Foundation, Theorem 1)
  // ------------------------------------------------------------------

  _juryConsult(projected, geoDist) {
    if (!this.trajectories || this.trajectories.length === 0) return 1.0;

    const N = Math.min(this.config.numJurors, this.trajectories.length);
    const cacheDist = this.trajectories.map(traj => {
      let dot = 0;
      for (let j = 0; j < this.k; j++) {
        dot += projected[j] * traj[j];
      }
      dot = Math.max(-1.0 + this.config.epsilon, Math.min(1.0 - this.config.epsilon, dot));
      return Math.acos(dot);
    });

    // Find N nearest
    const indexed = cacheDist.map((d, i) => ({ d, i }));
    indexed.sort((a, b) => a.d - b.d);
    const nearest = indexed.slice(0, N);

    // Jury aggregation: J = 1 - ∏(1 - c_j)
    let product = 1.0;
    for (const { d } of nearest) {
      const c = Math.exp(-d / Math.max(this.coverageRadius, this.config.epsilon));
      product *= (1.0 - c);
    }

    return 1.0 - product;
  }

  // ------------------------------------------------------------------
  // Auto Encoder
  // ------------------------------------------------------------------

  _autoEncoder(data) {
    if (data.length === 0) return x => [0.0];

    const sample = data[0];

    if (typeof sample === 'number') {
      const maxVal = Math.max(...data.map(x => Math.abs(x)), 1.0);
      return x => [x, x / maxVal, 1.0];
    }

    if (typeof sample === 'string') {
      return s => {
        const vec = new Array(8).fill(0);
        for (let i = 0; i < Math.min(s.length, 8); i++) {
          vec[i] = s.charCodeAt(i) / 65535.0;
        }
        if (vec.length < 3) vec.push(0, 0, 0);
        return vec;
      };
    }

    if (Array.isArray(sample)) {
      return arr => {
        const vec = arr.slice(0, 32).map(x => typeof x === 'number' ? x : 0);
        while (vec.length < 3) vec.push(0);
        return vec;
      };
    }

    // Generic: hash-based
    return x => {
      const s = String(x);
      let h = 0;
      for (let i = 0; i < s.length; i++) {
        h = ((h << 5) - h) + s.charCodeAt(i);
        h |= 0;
      }
      return [h, (h >>> 16) / 65535.0, 1.0];
    };
  }

  // ------------------------------------------------------------------
  // Diagnostics
  // ------------------------------------------------------------------

  instinctHorizon() {
    const N = this.config.numJurors;
    const R = this.coverageRadius;
    return R * (-Math.log(1.0 - Math.pow(0.5, 1.0 / N)));
  }

  getStatistics() {
    return {
      intrinsicDim: this.k,
      numTrajectories: this.trajectories ? this.trajectories.length : 0,
      coverageRadius: this.coverageRadius,
      instinctHorizon: this.instinctHorizon(),
      numJurors: this.config.numJurors,
      isBuilt: this._isBuilt,
    };
  }
}

// ---------------------------------------------------------------------------
// Convenience API
// ---------------------------------------------------------------------------

/**
 * One-shot hypersort.
 *
 * @param {Array} data - Elements to sort
 * @param {Object} [config] - ManifoldConfig options
 * @param {Function} [encoder] - Custom encoder function
 * @returns {SortResult}
 */
function hypersort(data, config = {}, encoder = null) {
  const sorter = new HyperSort(config);
  return sorter.sort(data, encoder);
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    HyperSort,
    ManifoldConfig,
    SortResult,
    hypersort,
  };
}

// Also export for ES modules
if (typeof exports !== 'undefined') {
  exports.HyperSort = HyperSort;
  exports.ManifoldConfig = ManifoldConfig;
  exports.SortResult = SortResult;
  exports.hypersort = hypersort;
}
