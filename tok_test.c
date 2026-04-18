/* Quick diagnostic: print what tok=106 and tok=107 decode to */
#include <stdio.h>
int llm_test_decode_token(int tok, char *out, int max_out);
int llm_load(const char *path);

/* Just print the decode */
int main(void) {
    char p[64];
    int n;
    /* We need the model loaded to test this... use a workaround */
    printf("Note: run geodessical with --decode-tokens 106 107 to test\n");
    return 0;
}
