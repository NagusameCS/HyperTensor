/* =============================================================================
 * TensorOS - Self-Test Framework Header
 * Built-in test infrastructure for production verification
 * =============================================================================*/

#ifndef TENSOROS_SELFTEST_H
#define TENSOROS_SELFTEST_H

/* Test assertion — records failure and returns 0 on fail */
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        selftest_fail(__func__, msg); \
        return 0; \
    } \
} while (0)

/* Float comparison with tolerance */
#define TEST_ASSERT_FLOAT_EQ(a, b, tol, msg) do { \
    float _a = (a), _b = (b), _t = (tol); \
    float _d = _a - _b; \
    if (_d < 0) _d = -_d; \
    if (_d > _t) { \
        selftest_fail_float(__func__, msg, _a, _b); \
        return 0; \
    } \
} while (0)

/* Internal: record a test failure */
void selftest_fail(const char *test_name, const char *msg);
void selftest_fail_float(const char *test_name, const char *msg, float got, float expected);

/* Run all self-tests and print summary */
void selftest_run_all(void);

#endif /* TENSOROS_SELFTEST_H */
