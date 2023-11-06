/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.example.util.ReaderSupplierFactory;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Test;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static java.lang.Math.PI;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class Test2DFloatVectors extends LuceneTestCase {
    @Test
    public void testThreshold() {
        var R = getRandom();
        // generate 2D vectors
        float[][] vectors = new float[10000][2];
        for (int i = 0; i < vectors.length; i++) {
            vectors[i][0] = R.nextFloat();
            vectors[i][1] = R.nextFloat();
        }

        var ravv = new ListRandomAccessVectorValues(List.of(vectors), 2);
        var builder = new GraphIndexBuilder<>(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN, 6, 32, 1.2f, 1.2f);
        var graph = builder.build();
        var searcher = new GraphSearcher.Builder<>(graph.getView()).build();

        for (int i = 0; i < 10; i++) {
            TestParams tp = createTestParams(vectors);

            NeighborSimilarity.ExactScoreFunction sf = j -> VectorSimilarityFunction.EUCLIDEAN.compare(tp.q, ravv.vectorValue(j));
            var result = searcher.search(sf, null, vectors.length, tp.th, Bits.ALL);

            assert result.getVisitedCount() < vectors.length : "visited all vectors for threshold " + tp.th;
            assert result.getNodes().length >= 0.9 * tp.exactCount : "returned " + result.getNodes().length + " nodes for threshold " + tp.th + " but should have returned at least " + tp.exactCount;
        }
    }

    // it's not an interesting test if all the vectors are within the threshold
    private TestParams createTestParams(float[][] vectors) {
        var R = getRandom();

        long exactCount;
        float[] q;
        float th;
        do {
            q = new float[]{R.nextFloat(), R.nextFloat()};
            th = (float) (0.2 + 0.8 * R.nextDouble());
            float[] finalQ = q;
            float finalTh = th;
            exactCount = Arrays.stream(vectors).filter(v -> VectorSimilarityFunction.EUCLIDEAN.compare(finalQ, v) >= finalTh).count();
        } while (!(exactCount < vectors.length * 0.8));

        return new TestParams(exactCount, q, th);
    }

    private static class TestParams {
        public final long exactCount;
        public final float[] q;
        public final float th;

        public TestParams(long exactCount, float[] q, float th) {
            this.exactCount = exactCount;
            this.q = q;
            this.th = th;
        }
    }

    /** creates a grid along the integer points (0..width, 0..width), so it will have width**2 points */
    private void test2DGrid(int width) throws IOException {
        // create vector grid
        var rawVectors = new ArrayList<float[]>(width * width);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < width; y++) {
                rawVectors.add(new float[] {x, y});
            }
        }
        var ravv = new ListRandomAccessVectorValues(rawVectors, 2);

        // build in-memory graph
        GraphIndexBuilder<float[]> builder = new GraphIndexBuilder<>(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN, 16, 100, 1.2f, 1.4f);
        var onHeapGraph = builder.build();
        System.out.printf("Avg degree is %.2f with %.2f short edges%n", onHeapGraph.getAverageDegree(), onHeapGraph.getAverageShortEdges());

        // quantize and compute on-disk graph
        System.out.println("Graph complete. Quantizing...");
        var pq = ProductQuantization.compute(ravv, ravv.dimension(), true);
        var encoded = pq.encodeAll(rawVectors);
        var cv = pq.createCompressedVectors(encoded);
        var graphPath = Files.createTempFile("diskann", "graph");
        try (var outputStream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(graphPath)))) {
            OnDiskGraphIndex.write(onHeapGraph, ravv, outputStream);
        }

        // random query vectors inside our grid (not restricted to integer points)
        var R = getRandom();
        var tRecall = 0.0;
        var qRecall = 0.0;
        int nQueries = 1000;
        for (int i = 0; i < nQueries; i++) {
            var q = new float[]{width * R.nextFloat(), width * R.nextFloat()};

            // compute the ground truth
            // the top 50 should be in a circle that is within the bounds of a square 20x20 around the query point
            // (10x10 would be enough but I want to simplify the treatment near edges)
            int topK = 50;
            var bounded = new ArrayList<float[]>();
            var upperLeft = new float[]{q[0] - 10, q[1] - 10};
            var lowerRight = new float[]{q[0] + 10, q[1] + 10};
            for (float[] v : rawVectors) {
                if (v[0] >= upperLeft[0] && v[0] <= lowerRight[0] && v[1] >= upperLeft[1] && v[1] <= lowerRight[1]) {
                    bounded.add(v);
                }
            }
            bounded.sort(Comparator.comparingDouble((float[] v) -> VectorSimilarityFunction.EUCLIDEAN.compare(q, v)).reversed());
            var gt = bounded.subList(0, topK);

            // compute recall against the uncompressed vectors
            var results = GraphSearcher.search(q, topK, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN, onHeapGraph, Bits.ALL);
            var nMatches = Arrays.stream(results.getNodes()).filter(n -> gt.contains(ravv.vectorValue(n.node))).count();
            var recall = (double) nMatches / topK;
            tRecall += recall;

            try (var onDiskGraph = new OnDiskGraphIndex<float[]>(ReaderSupplierFactory.open(graphPath), 0)) {
                var searcher = new GraphSearcher.Builder<>(onDiskGraph.getView()).build();
                NeighborSimilarity.ReRanker<float[]> rr = (j, vectors) -> VectorSimilarityFunction.EUCLIDEAN.compare(q, vectors.get(j));
                results = searcher.search(cv.approximateScoreFunctionFor(q, VectorSimilarityFunction.EUCLIDEAN), rr, topK, Bits.ALL);
                nMatches = Arrays.stream(results.getNodes()).filter(n -> gt.contains(ravv.vectorValue(n.node))).count();
                recall = (double) nMatches / topK;
                qRecall += recall;
            }
        }
        System.out.printf("Average raw/quantized recall %.2f%%/%2f%%%%n", 100.0 * tRecall / nQueries, 100.0 * qRecall / nQueries);
    }

    @Test
    public void test2DGrid() throws IOException {
//        System.out.println("### 10k vectors");
//        test2DGrid(100);

        // 49, 49, 52% recall with 1M vectors
        // 79, 84, 76% with 100k
        System.out.println("\n### 1M vectors");
        test2DGrid(200);
    }

    /** creates nPoints on the surface of a 1/8 unit sphere */
    private void test3DSphere(int nPoints) throws IOException {
        // create vector grid
        var rawVectors = new ArrayList<float[]>(nPoints);
        var divisor = sqrt(nPoints);
        var halfPi = PI / 2;
        for (var a = 0.0; a < halfPi; a += halfPi / divisor) {
            for (var b = 0.0; b < halfPi; b += halfPi / divisor) {
                rawVectors.add(new float[] {(float) (cos(a) * cos(b)), (float) (cos(a) * sin(b)), (float) sin(a)});
            }
        }
        var ravv = new ListRandomAccessVectorValues(rawVectors, rawVectors.get(0).length);

        // build in-memory graph
        GraphIndexBuilder<float[]> builder = new GraphIndexBuilder<>(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, 16, 100, 1.2f, 1.4f);
        var onHeapGraph = builder.build();
        // quantize and compute on-disk graph
        var pq = ProductQuantization.compute(ravv, ravv.dimension(), true);
        var encoded = pq.encodeAll(rawVectors);
        var cv = pq.createCompressedVectors(encoded);
        var graphPath = Files.createTempFile("diskann", "graph");
        try (var outputStream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(graphPath)))) {
            OnDiskGraphIndex.write(onHeapGraph, ravv, outputStream);
        }

        // random query vectors on our sphere
        var R = getRandom();
        for (int i = 0; i < 10; i++) {
            var a = R.nextDouble() * halfPi;
            var b = R.nextDouble() * halfPi;
            var q = new float[] {(float) (cos(a) * cos(b)), (float) (cos(a) * sin(b)), (float) sin(a)};

            // compute the ground truth
            // the top 50 should be within a cube that is 0.1 on a side, because I am bad at math
            // (this is a very loose bound but it's good enough that computing brute force inside it won't be terrible)
            int topK = 50;
            var bounded = new ArrayList<float[]>();
            var upperLeft = new float[]{q[0] - 0.1f, q[1] - 0.1f, q[2] - 0.1f};
            var lowerRight = new float[]{q[0] + 0.1f, q[1] + 0.1f, q[2] + 0.1f};
            for (float[] v : rawVectors) {
                if (v[0] >= upperLeft[0] && v[0] <= lowerRight[0] && v[1] >= upperLeft[1] && v[1] <= lowerRight[1] && v[2] >= upperLeft[2] && v[2] <= lowerRight[2]) {
                    bounded.add(v);
                }
            }
            bounded.sort(Comparator.comparingDouble((float[] v) -> VectorSimilarityFunction.EUCLIDEAN.compare(q, v)).reversed());
            var gt = bounded.subList(0, topK);
            assert gt.size() == topK;

            // compute recall against the uncompressed vectors
            var results = GraphSearcher.search(q, topK, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN, onHeapGraph, Bits.ALL);
            var nMatches = Arrays.stream(results.getNodes()).filter(n -> gt.contains(ravv.vectorValue(n.node))).count();
            System.out.printf("Uncompressed recall %.2f%%%n", 100.0 * nMatches / topK);

            try (var onDiskGraph = new OnDiskGraphIndex<float[]>(ReaderSupplierFactory.open(graphPath), 0)) {
                var searcher = new GraphSearcher.Builder<>(onDiskGraph.getView()).build();
                NeighborSimilarity.ReRanker<float[]> rr = (j, vectors) -> VectorSimilarityFunction.EUCLIDEAN.compare(q, vectors.get(j));
                results = searcher.search(cv.approximateScoreFunctionFor(q, VectorSimilarityFunction.EUCLIDEAN), rr, topK, Bits.ALL);
                nMatches = Arrays.stream(results.getNodes()).filter(n -> gt.contains(ravv.vectorValue(n.node))).count();
                System.out.printf("Quantized recall %.2f%%%n", 100.0 * nMatches / topK);
            }
        }
    }

    @Test
    public void test3DGrid() throws IOException {
        System.out.println("### 10k vectors");
        test3DSphere(10_000);

        System.out.println("\n### 1M vectors");
        test3DSphere(1_000_000);
    }
}
