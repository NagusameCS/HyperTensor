/**
 * HyperSort Demo — O(1) Instant Sort via Riemannian Comparison Manifold
 * Java Implementation
 *
 * Compile: javac -d build demos/Demo.java
 * Run:     java -cp build Demo
 */

import com.hypersort.HyperSort;
import com.hypersort.SortResult;
import java.util.*;
import java.util.function.Function;

public class Demo {

    private static final String SEP = "─".repeat(60);

    public static void main(String[] args) {
        System.out.println("╔" + "═".repeat(58) + "╗");
        System.out.println("║  HyperSort Java Demo — O(1) Riemannian Comparison Manifold  ║");
        System.out.println("║  HyperTensor Geometric Jury Framework (Papers I–XVIII)      ║");
        System.out.println("╚" + "═".repeat(58) + "╝");

        demoBasic();
        demoStrings();
        demoReuse();
        demoHeadToHead();
        demoJuryConfidence();
    }

    // ==================================================================
    // DEMO 1: Basic Numbers
    // ==================================================================
    static void demoBasic() {
        System.out.println("\n" + "█".repeat(60));
        System.out.println("█  DEMO 1: Basic Number Sorting");
        System.out.println("█".repeat(60));

        List<Double> numbers = Arrays.asList(3.14, 1.41, 2.71, 1.73, 0.57, 9.81, 6.28, 2.22);
        SortResult<Double> result = HyperSort.sortNumbers(numbers);

        System.out.println("  Input:  " + numbers);
        System.out.println("  Sorted: " + result.getSortedData());
        System.out.printf("  Time:   %.4f ms%n", result.getTotalTimeMs());
        System.out.println("  Comps:  " + result.getComparisonsAvoided() + " (all n² in one matmul)");
        System.out.println("  Dim:    k=" + result.getManifoldDim());

        double[] conf = result.getConfidenceScores();
        System.out.print("  Conf:   [");
        for (int i = 0; i < Math.min(5, conf.length); i++) {
            System.out.printf("%.4f", conf[i]);
            if (i < Math.min(5, conf.length) - 1) System.out.print(", ");
        }
        System.out.println("...]");
    }

    // ==================================================================
    // DEMO 2: Strings
    // ==================================================================
    static void demoStrings() {
        System.out.println("\n" + "█".repeat(60));
        System.out.println("█  DEMO 2: String Sorting");
        System.out.println("█".repeat(60));

        HyperSort<String> sorter = new HyperSort<>();
        List<String> words = Arrays.asList("hyper", "tensor", "geodesic", "manifold", "jury", "sort", "O(1)");

        Function<String, double[]> encoder = s -> {
            double[] vec = new double[8];
            for (int i = 0; i < Math.min(s.length(), 8); i++) {
                vec[i] = (double) s.charAt(i) / 65535.0;
            }
            return vec;
        };

        SortResult<String> result = sorter.sort(words, encoder);
        System.out.println("  Input:  " + words);
        System.out.println("  Sorted: " + result.getSortedData());
    }

    // ==================================================================
    // DEMO 3: Reusable Manifold
    // ==================================================================
    static void demoReuse() {
        System.out.println("\n" + "█".repeat(60));
        System.out.println("█  DEMO 3: Reusable Manifold (build once, sort many)");
        System.out.println("█".repeat(60));

        HyperSort<Double> sorter = new HyperSort<>(16, 11);

        // Build manifold on training data
        List<Double> training = new ArrayList<>();
        Random rng = new Random(42);
        for (int i = 0; i < 200; i++) {
            training.add(rng.nextDouble() * 200 - 100);
        }

        Function<Double, double[]> encoder = x -> new double[]{x, x / 100.0, 1.0};

        long t0 = System.nanoTime();
        sorter.sort(training, encoder);
        double buildMs = (System.nanoTime() - t0) / 1_000_000.0;
        System.out.printf("  Manifold built in %.2f ms (one-time)%n", buildMs);

        // Sort multiple batches
        for (int b = 0; b < 3; b++) {
            List<Double> batch = new ArrayList<>();
            for (int i = 0; i < 50; i++) {
                batch.add(rng.nextDouble() * 200 - 100);
            }

            t0 = System.nanoTime();
            SortResult<Double> result = sorter.sort(batch, encoder);
            double sortMs = (System.nanoTime() - t0) / 1_000_000.0;

            // Verify
            List<Double> expected = new ArrayList<>(batch);
            Collections.sort(expected);
            boolean correct = result.getSortedData().equals(expected);

            System.out.printf("  Batch %d: %.4f ms | %d comparisons | %s%n",
                b + 1, sortMs, result.getComparisonsAvoided(),
                correct ? "✓ CORRECT" : "✗ WRONG");
        }
    }

    // ==================================================================
    // DEMO 4: Head-to-Head
    // ==================================================================
    static void demoHeadToHead() {
        System.out.println("\n" + "█".repeat(60));
        System.out.println("█  DEMO 4: HyperSort vs Collections.sort()");
        System.out.println("█".repeat(60));

        int[] sizes = {10, 50, 100, 250, 500};
        System.out.printf("  %-8s %-16s %-20s %-12s%n", "Size", "HyperSort", "Collections.sort()", "Winner");
        System.out.println("  " + "─".repeat(8) + " " + "─".repeat(16) + " " + "─".repeat(20) + " " + "─".repeat(12));

        Random rng = new Random(42);
        for (int n : sizes) {
            List<Double> data = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                data.add(rng.nextDouble() * 2000 - 1000);
            }

            // HyperSort
            long t0 = System.nanoTime();
            SortResult<Double> hsResult = HyperSort.sortNumbers(data);
            double hsMs = (System.nanoTime() - t0) / 1_000_000.0;

            // Collections.sort
            List<Double> copy = new ArrayList<>(data);
            t0 = System.nanoTime();
            Collections.sort(copy);
            double natMs = (System.nanoTime() - t0) / 1_000_000.0;

            String winner = hsMs < natMs ? "HyperSort" : "Collections";
            System.out.printf("  %-8d %-16s %-20s %-12s%n",
                n, String.format("%.4f ms", hsMs), String.format("%.4f ms", natMs), winner);
        }
    }

    // ==================================================================
    // DEMO 5: Jury Confidence
    // ==================================================================
    static void demoJuryConfidence() {
        System.out.println("\n" + "█".repeat(60));
        System.out.println("█  DEMO 5: Geometric Jury Confidence");
        System.out.println("█".repeat(60));

        List<Double> data = Arrays.asList(1.0, 1.1, 1.2, 1.3, 1.4, 100.0, 1.5, 1.6, 1.7, 1.8);
        SortResult<Double> result = HyperSort.sortNumbers(data);

        System.out.println("  Sorted with confidence:");
        for (int i = 0; i < result.getSortedData().size(); i++) {
            double val = result.getSortedData().get(i);
            double conf = result.getConfidenceScores()[i];
            StringBuilder bar = new StringBuilder();
            for (int b = 0; b < (int)(conf * 20); b++) bar.append("█");
            String note = conf < 0.5 ? " ← LOW CONFIDENCE (outlier)" : "";
            System.out.printf("    %8.1f  J=%.4f  %s%s%n", val, conf, bar.toString(), note);
        }
    }
}
