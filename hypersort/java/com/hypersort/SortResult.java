package com.hypersort;

import java.util.Collections;
import java.util.List;

/**
 * Result of a HyperSort sorting operation with metadata.
 */
public class SortResult<T> {
    private final List<T> sortedData;
    private final int[] originalIndices;
    private final double[] confidenceScores;
    private final double totalTimeMs;
    private final int manifoldDim;
    private final long comparisonsAvoided;

    public SortResult(List<T> sortedData, int[] originalIndices,
                      double[] confidenceScores, double totalTimeMs,
                      int manifoldDim, long comparisonsAvoided) {
        this.sortedData = Collections.unmodifiableList(sortedData);
        this.originalIndices = originalIndices.clone();
        this.confidenceScores = confidenceScores.clone();
        this.totalTimeMs = totalTimeMs;
        this.manifoldDim = manifoldDim;
        this.comparisonsAvoided = comparisonsAvoided;
    }

    public List<T> getSortedData() { return sortedData; }
    public int[] getOriginalIndices() { return originalIndices; }
    public double[] getConfidenceScores() { return confidenceScores; }
    public double getTotalTimeMs() { return totalTimeMs; }
    public int getManifoldDim() { return manifoldDim; }
    public long getComparisonsAvoided() { return comparisonsAvoided; }
}
