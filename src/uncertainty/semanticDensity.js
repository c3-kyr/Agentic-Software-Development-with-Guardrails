/**
 * Semantic Density / Embedding Variance Score
 *
 * U_semantic = (1/N) * Σ ||v_i - v̄||²
 *
 * Also performs cluster detection to identify competing architectural plans.
 */

/**
 * Compute the centroid of a set of vectors
 */
function centroid(vectors) {
    const dim = vectors[0].length;
    const c = new Array(dim).fill(0);
    for (const v of vectors) {
        for (let i = 0; i < dim; i++) {
            c[i] += v[i];
        }
    }
    return c.map((x) => x / vectors.length);
}

/**
 * Squared L2 distance between two vectors
 */
function sqDist(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

/**
 * L2 distance
 */
function dist(a, b) {
    return Math.sqrt(sqDist(a, b));
}

/**
 * Simple k-means clustering
 */
function kmeans(vectors, k, maxIter = 50) {
    const n = vectors.length;
    const dim = vectors[0].length;

    // Initialize centroids using k-means++ lite
    const centroids = [vectors[0].slice()];
    for (let c = 1; c < k; c++) {
        let maxDist = -1;
        let bestIdx = 0;
        for (let i = 0; i < n; i++) {
            const minD = Math.min(...centroids.map((ctr) => dist(vectors[i], ctr)));
            if (minD > maxDist) {
                maxDist = minD;
                bestIdx = i;
            }
        }
        centroids.push(vectors[bestIdx].slice());
    }

    let assignments = new Array(n).fill(0);

    for (let iter = 0; iter < maxIter; iter++) {
        // Assign
        const newAssign = vectors.map((v) => {
            let bestK = 0;
            let bestD = Infinity;
            for (let c = 0; c < k; c++) {
                const d = sqDist(v, centroids[c]);
                if (d < bestD) {
                    bestD = d;
                    bestK = c;
                }
            }
            return bestK;
        });

        // Check convergence
        if (newAssign.every((a, i) => a === assignments[i])) break;
        assignments = newAssign;

        // Update centroids
        for (let c = 0; c < k; c++) {
            const members = vectors.filter((_, i) => assignments[i] === c);
            if (members.length > 0) {
                const newC = centroid(members);
                for (let d = 0; d < dim; d++) centroids[c][d] = newC[d];
            }
        }
    }

    return { assignments, centroids };
}

/**
 * Compute silhouette score for a clustering
 */
function silhouetteScore(vectors, assignments, k) {
    const n = vectors.length;
    if (k === 1 || n <= k) return 0;

    let totalSil = 0;
    for (let i = 0; i < n; i++) {
        const ci = assignments[i];

        // a(i) = mean distance to same cluster
        const sameCluster = vectors.filter((_, j) => j !== i && assignments[j] === ci);
        const a = sameCluster.length > 0
            ? sameCluster.reduce((s, v) => s + dist(vectors[i], v), 0) / sameCluster.length
            : 0;

        // b(i) = min mean distance to any other cluster
        let b = Infinity;
        for (let c = 0; c < k; c++) {
            if (c === ci) continue;
            const otherCluster = vectors.filter((_, j) => assignments[j] === c);
            if (otherCluster.length === 0) continue;
            const meanD = otherCluster.reduce((s, v) => s + dist(vectors[i], v), 0) / otherCluster.length;
            b = Math.min(b, meanD);
        }

        if (b === Infinity) b = 0;
        const sil = Math.max(a, b) > 0 ? (b - a) / Math.max(a, b) : 0;
        totalSil += sil;
    }

    return totalSil / n;
}

/**
 * Compute Semantic Density Score
 *
 * @param {number[][]} embeddings - Array of embedding vectors
 * @returns {{ score: number, clusterCount: number, clusterSizes: number[] }}
 */
export function computeSemanticDensity(embeddings) {
    if (!embeddings || embeddings.length === 0) {
        return { score: 0, clusterCount: 1, clusterSizes: [0] };
    }

    const N = embeddings.length;
    const c = centroid(embeddings);

    // U_semantic = (1/N) * Σ ||v_i - v̄||²
    let variance = 0;
    for (const v of embeddings) {
        variance += sqDist(v, c);
    }
    variance /= N;

    // Normalize to [0, 1] range — clamp at reasonable max
    // For unit-normalized embeddings in dim=8, max variance ≈ 4.0
    const normalizedScore = Math.min(variance / 2.0, 1.0);

    // Cluster detection: try k=1,2,3 — pick best silhouette
    let bestK = 1;
    let bestSil = -1;

    for (let k = 2; k <= Math.min(3, N - 1); k++) {
        const { assignments } = kmeans(embeddings, k);
        const sil = silhouetteScore(embeddings, assignments, k);
        if (sil > bestSil + 0.1) {
            // Require meaningful improvement
            bestSil = sil;
            bestK = k;
        }
    }

    // Count cluster sizes
    const { assignments } = bestK > 1 ? kmeans(embeddings, bestK) : { assignments: new Array(N).fill(0) };
    const clusterSizes = [];
    for (let k = 0; k < bestK; k++) {
        clusterSizes.push(assignments.filter((a) => a === k).length);
    }

    return {
        score: normalizedScore,
        clusterCount: bestK,
        clusterSizes: clusterSizes.sort((a, b) => b - a),
    };
}
