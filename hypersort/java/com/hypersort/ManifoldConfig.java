package com.hypersort;

/**
 * Configuration for the Comparison Manifold.
 */
public class ManifoldConfig {
    int intrinsicDim = 32;
    int numJurors = 7;
    double coverageRadius = 1.0;
    double temperature = 8.0;
    double epsilon = 1e-8;
    double cacheThreshold = 0.05;

    public ManifoldConfig() {}

    public ManifoldConfig(int intrinsicDim, int numJurors) {
        this.intrinsicDim = intrinsicDim;
        this.numJurors = numJurors;
    }
}
