/**
 * Tests for HyperSort JavaScript implementation.
 *
 * Run with: node tests/test_hypersort.js
 */

const { HyperSort, hypersort, ManifoldConfig, SortResult } = require('../javascript/hypersort.js');

// ---------------------------------------------------------------------------
// Simple Test Framework
// ---------------------------------------------------------------------------

let passed = 0;
let failed = 0;

function assert(condition, message) {
    if (condition) {
        passed++;
    } else {
        failed++;
        console.error(`FAIL: ${message}`);
    }
}

function assertDeepEqual(actual, expected, message) {
    const actualStr = JSON.stringify(actual);
    const expectedStr = JSON.stringify(expected);
    if (actualStr === expectedStr) {
        passed++;
    } else {
        failed++;
        console.error(`FAIL: ${message}`);
        console.error(`  Expected: ${expectedStr}`);
        console.error(`  Actual:   ${actualStr}`);
    }
}

// ---------------------------------------------------------------------------
// Basic Tests
// ---------------------------------------------------------------------------

function testEmptyList() {
    const result = hypersort([]);
    assert(result.sortedData.length === 0, 'Empty list should return empty');
}

function testSingleElement() {
    const result = hypersort([42]);
    assert(result.sortedData[0] === 42, 'Single element preserved');
    assert(result.confidenceScores[0] > 0, 'Confidence should be positive');
}

function testSortedNumbers() {
    const result = hypersort([1, 2, 3, 4, 5]);
    assertDeepEqual(result.sortedData, [1, 2, 3, 4, 5], 'Already sorted stays sorted');
}

function testReverseNumbers() {
    const result = hypersort([5, 4, 3, 2, 1]);
    assertDeepEqual(result.sortedData, [1, 2, 3, 4, 5], 'Reverse sorted correctly');
}

function testRandomNumbers() {
    const data = [3.14, 1.41, 2.71, 1.73, 0.57];
    const result = hypersort(data);
    const expected = [...data].sort((a, b) => a - b);
    assertDeepEqual(result.sortedData, expected, 'Random numbers sorted correctly');
}

function testDuplicates() {
    const result = hypersort([5, 3, 5, 1, 3, 1]);
    assertDeepEqual(result.sortedData, [1, 1, 3, 3, 5, 5], 'Duplicates handled');
}

function testStrings() {
    const result = hypersort(['banana', 'apple', 'cherry', 'date']);
    const expected = [...result.sortedData].sort();
    assertDeepEqual(result.sortedData, expected, 'Strings sorted');
}

function testMixedIntegers() {
    const data = [7, 2, 9, 1, 5, 3, 8, 4, 6, 0];
    const result = hypersort(data);
    assertDeepEqual(result.sortedData, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], '0-9 sorted');
}

// ---------------------------------------------------------------------------
// Metadata Tests
// ---------------------------------------------------------------------------

function testResultStructure() {
    const result = hypersort([3, 1, 2]);
    assert(result instanceof SortResult, 'Result is SortResult');
    assert(result.sortedData !== null, 'sortedData exists');
    assert(result.originalIndices.length === 3, 'originalIndices length correct');
    assert(result.confidenceScores.length === 3, 'confidenceScores length correct');
    assert(result.totalTimeMs >= 0, 'totalTimeMs non-negative');
    assert(result.manifoldDim > 0, 'manifoldDim positive');
    assert(result.comparisonsAvoided > 0, 'comparisonsAvoided positive');
}

function testConfidenceRange() {
    const result = hypersort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]);
    for (const conf of result.confidenceScores) {
        assert(conf >= 0.0 && conf <= 1.0, `Confidence ${conf} in [0,1]`);
    }
}

// ---------------------------------------------------------------------------
// Configuration Tests
// ---------------------------------------------------------------------------

function testCustomConfig() {
    const sorter = new HyperSort({ intrinsicDim: 64, numJurors: 11 });
    const result = sorter.sort([3, 1, 2]);
    assert(result.manifoldDim <= 64, 'Respects intrinsicDim');
    assert(result.sortedData[0] === 1, 'Sorts correctly with custom config');
}

function testStatistics() {
    const sorter = new HyperSort();
    const stats = sorter.getStatistics();
    assert(typeof stats.intrinsicDim === 'number', 'Stats has intrinsicDim');
    assert(typeof stats.coverageRadius === 'number', 'Stats has coverageRadius');
    assert(typeof stats.instinctHorizon === 'number', 'Stats has instinctHorizon');
    assert(stats.isBuilt === false, 'Not built initially');

    sorter.sort([1, 2, 3]);
    const stats2 = sorter.getStatistics();
    assert(stats2.isBuilt === true, 'Built after sort');
}

// ---------------------------------------------------------------------------
// Custom Encoder Tests
// ---------------------------------------------------------------------------

function testCustomEncoder() {
    const data = ['ccc', 'a', 'bb'];
    const encoder = s => [s.length, s.charCodeAt(0) || 0, (s.charCodeAt(s.length - 1)) || 0];
    const result = hypersort(data, {}, encoder);
    assert(result.sortedData.length === 3, 'Custom encoder works');
}

// ---------------------------------------------------------------------------
// Run All Tests
// ---------------------------------------------------------------------------

console.log('=== HyperSort JavaScript Tests ===\n');

testEmptyList();
testSingleElement();
testSortedNumbers();
testReverseNumbers();
testRandomNumbers();
testDuplicates();
testStrings();
testMixedIntegers();
testResultStructure();
testConfidenceRange();
testCustomConfig();
testStatistics();
testCustomEncoder();

console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);

if (failed > 0) {
    process.exit(1);
}
