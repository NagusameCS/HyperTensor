/* =============================================================================
 * TensorOS - AI Shell Implementation
 * Full-featured interactive shell with real kernel API integration
 * =============================================================================*/

#include "userland/shell/aishell.h"
#include "kernel/core/perf.h"
#include "kernel/core/cpu_features.h"
#include "kernel/core/smp.h"
#include "kernel/mm/tensor_mm.h"
#include "kernel/fs/tensorfs.h"
#include "kernel/fs/git.h"
#include "kernel/sched/tensor_sched.h"
#include "kernel/drivers/blk/virtio_blk.h"
#include "kernel/net/netstack.h"
#include "kernel/security/ssh.h"
#include "kernel/security/security.h"
#include "kernel/update/ota.h"
#include "runtime/nn/llm.h"
#include "runtime/jit/x86_jit.h"

/* ---- Forward declarations ---- */
static int  shell_exec_builtin(aishell_t *sh, int argc, char **argv);
static void shell_parse_line(const char *line, int *argc, char *argv[]);
static void shell_print_banner(void);
static void shell_print_help(void);

/* ---- Helpers ---- */

static int shell_strcmp(const char *a, const char *b)
{
    while (*a && *a == *b) { a++; b++; }
    return *(unsigned char *)a - *(unsigned char *)b;
}

__attribute__((unused))
static int shell_strncmp(const char *a, const char *b, uint64_t n)
{
    while (n && *a && *a == *b) { a++; b++; n--; }
    return n == 0 ? 0 : *(unsigned char *)a - *(unsigned char *)b;
}

static uint64_t shell_strlen(const char *s)
{
    uint64_t n = 0;
    while (s[n]) n++;
    return n;
}

static void shell_strcpy(char *dst, const char *src)
{
    while (*src) *dst++ = *src++;
    *dst = 0;
}

static void shell_strncpy(char *dst, const char *src, uint64_t size)
{
    if (!size) return;
    uint64_t i = 0;
    while (i < size - 1 && src[i]) { dst[i] = src[i]; i++; }
    dst[i] = 0;
}

static int shell_atoi(const char *s)
{
    int neg = 0, val = 0;
    if (*s == '-') { neg = 1; s++; }
    while (*s >= '0' && *s <= '9') { val = val * 10 + (*s - '0'); s++; }
    return neg ? -val : val;
}

static uint64_t shell_atou64(const char *s)
{
    uint64_t val = 0;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
        while (1) {
            char c = *s++;
            if (c >= '0' && c <= '9') val = val * 16 + (c - '0');
            else if (c >= 'a' && c <= 'f') val = val * 16 + (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') val = val * 16 + (c - 'A' + 10);
            else break;
        }
    } else {
        while (*s >= '0' && *s <= '9') { val = val * 10 + (*s - '0'); s++; }
    }
    return val;
}

/* =============================================================================
 * Key Reading — decodes PS/2 + VT100 escape sequences into KEY_* constants
 * =============================================================================*/

/* Check if a character is available (keyboard ring buffer or serial) */
static int shell_key_available(void)
{
    if (keyboard_has_key()) return 1;
#ifndef __aarch64__
    if (inb(0x3F8 + 5) & 0x01) return 1;  /* COM1 data ready */
#endif
    return 0;
}

/* Brief spin-wait for escape sequence bytes (serial needs ~87us/byte @115200) */
static int shell_wait_key(int iterations)
{
    for (int i = 0; i < iterations; i++) {
        if (shell_key_available()) return 1;
        for (volatile int j = 0; j < 200; j++);
    }
    return shell_key_available();
}

/* Read a single key, decoding ESC [ sequences into KEY_* constants */
static int shell_read_key(void)
{
    char c = keyboard_getchar();
    if (c == 0) return 0;

    /* ESC sequence: \x1b[... */
    if (c == 0x1B) {
        if (!shell_wait_key(80)) return 0x1B;  /* bare ESC */
        char c2 = keyboard_getchar();
        if (c2 == '[') {
            if (!shell_wait_key(80)) return 0x1B;
            char c3 = keyboard_getchar();
            switch (c3) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
                case 'H': return KEY_HOME;
                case 'F': return KEY_END;
                case '1': if (shell_wait_key(40)) { char t = keyboard_getchar(); if (t == '~') return KEY_HOME; } return KEY_HOME;
                case '2': if (shell_wait_key(40)) keyboard_getchar(); return KEY_INSERT;
                case '3': if (shell_wait_key(40)) keyboard_getchar(); return KEY_DELETE;
                case '4': if (shell_wait_key(40)) keyboard_getchar(); return KEY_END;
                case '5': if (shell_wait_key(40)) keyboard_getchar(); return KEY_PGUP;
                case '6': if (shell_wait_key(40)) keyboard_getchar(); return KEY_PGDN;
            }
        } else if (c2 == 'O') {
            if (shell_wait_key(40)) {
                char c3 = keyboard_getchar();
                if (c3 == 'H') return KEY_HOME;
                if (c3 == 'F') return KEY_END;
            }
        }
        return 0x1B;
    }

    return (int)(unsigned char)c;
}

/* =============================================================================
 * Command History
 * =============================================================================*/

static void history_add(shell_history_t *h, const char *line)
{
    if (shell_strlen(line) == 0) return;
    /* Skip duplicates of the most recent entry */
    if (h->count > 0) {
        int prev = (h->count - 1) % SHELL_MAX_HISTORY;
        if (shell_strcmp(h->lines[prev], line) == 0) return;
    }
    int idx = h->count % SHELL_MAX_HISTORY;
    shell_strncpy(h->lines[idx], line, SHELL_MAX_LINE);
    h->count++;
    h->cursor = h->count;
}

static const char *history_get(shell_history_t *h, int pos)
{
    int oldest = h->count > SHELL_MAX_HISTORY ? h->count - SHELL_MAX_HISTORY : 0;
    if (pos < oldest || pos >= h->count) return 0;
    return h->lines[pos % SHELL_MAX_HISTORY];
}

/* =============================================================================
 * Tab Completion
 * =============================================================================*/

static const char *shell_cmd_names[] = {
    "help", "status", "sysinfo", "uname", "uptime", "clear", "reboot",
    "exit", "quit",
    "ls", "dir", "cat", "stat", "mkdir", "touch", "write", "rm",
    "pwd", "cd", "cp", "mv", "head", "tail", "wc", "grep", "find",
    "df", "du", "tree",
    "cpu", "mem", "disk", "smp", "net", "ps", "lspci", "monitor",
    "sched", "jit", "free", "top", "dmesg", "irq",
    "llm", "ai", "reset", "infer", "train", "deploy", "serve", "api",
    "model", "tensor",
    "hexdump", "xxd", "calc", "time", "echo", "history", "selftest", "test",
    "peek", "poke", "strings", "crc32",
    "rdmsr", "wrmsr", "cpuid", "ioread", "iowrite",
    "repeat", "alias", "unalias", "set", "env", "export",
    "bench", "demo",
    "git", "pkg", "sandbox",
    "run", "ota", "flash",
    "whoami", "hostname", "date", "sleep", "true", "false",
    "yes", "seq", "rand", "logo", "version", "ver", "about",
    "ping", "ifconfig", "netstat",
    "sshd", "users", "passwd", "firewall", "fw", "audit",
    "keystore", "integrity", "sec-init", "sha256",
    "panic", "halt",
    0
};

static int shell_tab_complete(char *buf, int *len, int *cursor)
{
    /* Only complete the first token (command name) */
    for (int i = 0; i < *cursor; i++)
        if (buf[i] == ' ') return 0;  /* cursor past first word */

    int prefix_len = *cursor;
    if (prefix_len == 0) return 0;

    const char *matches[32];
    int match_count = 0;

    for (int i = 0; shell_cmd_names[i]; i++) {
        int ok = 1;
        for (int j = 0; j < prefix_len; j++) {
            if (shell_cmd_names[i][j] != buf[j]) { ok = 0; break; }
        }
        if (ok && match_count < 32)
            matches[match_count++] = shell_cmd_names[i];
    }

    if (match_count == 0) return 0;

    if (match_count == 1) {
        const char *m = matches[0];
        int mlen = (int)shell_strlen(m);
        for (int i = prefix_len; i < mlen && *len < SHELL_MAX_LINE - 2; i++) {
            for (int j = *len; j > *cursor; j--) buf[j] = buf[j - 1];
            buf[*cursor] = m[i];
            (*len)++; (*cursor)++;
        }
        if (*len < SHELL_MAX_LINE - 2) {
            for (int j = *len; j > *cursor; j--) buf[j] = buf[j - 1];
            buf[*cursor] = ' ';
            (*len)++; (*cursor)++;
        }
        buf[*len] = '\0';
        return 1;
    }

    /* Multiple matches: extend to longest common prefix */
    int common = (int)shell_strlen(matches[0]);
    for (int i = 1; i < match_count; i++) {
        int j = 0;
        while (j < common && matches[i][j] == matches[0][j]) j++;
        common = j;
    }
    if (common > prefix_len) {
        for (int i = prefix_len; i < common && *len < SHELL_MAX_LINE - 2; i++) {
            for (int j = *len; j > *cursor; j--) buf[j] = buf[j - 1];
            buf[*cursor] = matches[0][i];
            (*len)++; (*cursor)++;
        }
        buf[*len] = '\0';
        return 1;
    }

    /* Show all matches */
    kprintf("\n");
    for (int i = 0; i < match_count; i++) {
        kprintf("%-16s", matches[i]);
        if ((i + 1) % 5 == 0) kprintf("\n");
    }
    if (match_count % 5 != 0) kprintf("\n");
    return 2;  /* caller redraws prompt */
}

/* =============================================================================
 * Line Editor — Full readline-like editing with history, cursor, kill-ring
 *
 * Keybindings:
 *   Up/Down       Navigate command history
 *   Left/Right    Move cursor within line
 *   Home/End      Jump to start/end of line
 *   Ctrl+A/E      Same as Home/End
 *   Ctrl+K        Kill from cursor to end of line
 *   Ctrl+U        Kill from start of line to cursor
 *   Ctrl+W        Kill previous word
 *   Ctrl+T        Transpose characters
 *   Ctrl+L        Clear screen, redraw line
 *   Ctrl+C        Cancel current line
 *   Ctrl+D        Delete at cursor / EOF on empty line
 *   Tab           Auto-complete command names
 *   Delete        Delete char at cursor
 *   PgUp/PgDn     Jump to oldest/newest history
 * =============================================================================*/

static void line_refresh(const char *prompt, const char *buf, int len,
                         int cursor, int old_len)
{
    kprintf("\r%s", prompt);
    for (int i = 0; i < len; i++) {
        char ch[2] = {buf[i], 0};
        kprintf("%s", ch);
    }
    int extra = old_len > len ? old_len - len : 0;
    for (int i = 0; i < extra; i++) kprintf(" ");
    int back = (len - cursor) + extra;
    for (int i = 0; i < back; i++) kprintf("\b");
}

static int shell_read_line(aishell_t *sh, char *buf, int max)
{
    int len = 0, cursor = 0;
    int hist_nav = sh->history.count;
    static char saved_line[SHELL_MAX_LINE];
    int saved_len = 0;
    buf[0] = '\0';

    while (1) {
        int key = shell_read_key();
        if (key == 0) continue;

        /* Enter */
        if (key == '\n' || key == '\r') {
            buf[len] = '\0';
            kprintf("\n");
            return len;
        }

        /* Ctrl+C: cancel */
        if (key == 3) { kprintf("^C\n"); buf[0] = '\0'; return 0; }

        /* Ctrl+D: delete at cursor or EOF */
        if (key == 4) {
            if (len == 0) { shell_strncpy(buf, "exit", SHELL_MAX_LINE); kprintf("exit\n"); return 4; }
            if (cursor < len) {
                int ol = len;
                for (int i = cursor; i < len - 1; i++) buf[i] = buf[i + 1];
                len--; buf[len] = '\0';
                line_refresh(sh->prompt, buf, len, cursor, ol);
            }
            continue;
        }

        /* Ctrl+L: clear + redraw */
        if (key == 12) {
            vga_init();
            kprintf("%s", sh->prompt);
            for (int i = 0; i < len; i++) { char ch[2] = {buf[i], 0}; kprintf("%s", ch); }
            for (int i = len; i > cursor; i--) kprintf("\b");
            continue;
        }

        /* Ctrl+A / Home */
        if (key == 1 || key == KEY_HOME) {
            for (; cursor > 0; cursor--) kprintf("\b");
            continue;
        }

        /* Ctrl+E / End */
        if (key == 5 || key == KEY_END) {
            for (; cursor < len; cursor++) { char ch[2] = {buf[cursor], 0}; kprintf("%s", ch); }
            continue;
        }

        /* Ctrl+K: kill to end */
        if (key == 11) {
            int e = len - cursor;
            for (int i = 0; i < e; i++) kprintf(" ");
            for (int i = 0; i < e; i++) kprintf("\b");
            len = cursor; buf[len] = '\0';
            continue;
        }

        /* Ctrl+U: kill to start */
        if (key == 21) {
            int ol = len;
            for (int i = 0; i < len - cursor; i++) buf[i] = buf[cursor + i];
            len -= cursor; cursor = 0; buf[len] = '\0';
            line_refresh(sh->prompt, buf, len, cursor, ol);
            continue;
        }

        /* Ctrl+W: kill previous word */
        if (key == 23) {
            if (cursor == 0) continue;
            int oc = cursor, ol = len;
            while (cursor > 0 && buf[cursor - 1] == ' ') cursor--;
            while (cursor > 0 && buf[cursor - 1] != ' ') cursor--;
            int del = oc - cursor;
            for (int i = cursor; i + del < len; i++) buf[i] = buf[i + del];
            len -= del; buf[len] = '\0';
            line_refresh(sh->prompt, buf, len, cursor, ol);
            continue;
        }

        /* Ctrl+T: transpose chars */
        if (key == 20) {
            if (cursor > 0 && cursor < len) {
                char tmp = buf[cursor - 1]; buf[cursor - 1] = buf[cursor]; buf[cursor] = tmp;
                cursor++;
                line_refresh(sh->prompt, buf, len, cursor, len);
            }
            continue;
        }

        /* Tab: auto-complete */
        if (key == '\t') {
            int r = shell_tab_complete(buf, &len, &cursor);
            if (r == 2) {
                kprintf("%s", sh->prompt);
                for (int i = 0; i < len; i++) { char ch[2] = {buf[i], 0}; kprintf("%s", ch); }
                for (int i = len; i > cursor; i--) kprintf("\b");
            } else if (r == 1) {
                line_refresh(sh->prompt, buf, len, cursor, len);
            }
            continue;
        }

        /* Backspace */
        if (key == '\b' || key == 0x7F) {
            if (cursor > 0) {
                int ol = len;
                for (int i = cursor - 1; i < len - 1; i++) buf[i] = buf[i + 1];
                cursor--; len--; buf[len] = '\0';
                line_refresh(sh->prompt, buf, len, cursor, ol);
            }
            continue;
        }

        /* Delete */
        if (key == KEY_DELETE) {
            if (cursor < len) {
                int ol = len;
                for (int i = cursor; i < len - 1; i++) buf[i] = buf[i + 1];
                len--; buf[len] = '\0';
                line_refresh(sh->prompt, buf, len, cursor, ol);
            }
            continue;
        }

        /* Left */
        if (key == KEY_LEFT) {
            if (cursor > 0) { cursor--; kprintf("\b"); }
            continue;
        }

        /* Right */
        if (key == KEY_RIGHT) {
            if (cursor < len) { char ch[2] = {buf[cursor], 0}; kprintf("%s", ch); cursor++; }
            continue;
        }

        /* Up: history previous */
        if (key == KEY_UP) {
            int oldest = sh->history.count > SHELL_MAX_HISTORY ? sh->history.count - SHELL_MAX_HISTORY : 0;
            if (hist_nav <= oldest) continue;
            if (hist_nav == sh->history.count) {
                for (int i = 0; i <= len; i++) saved_line[i] = buf[i];
                saved_len = len;
            }
            hist_nav--;
            const char *hl = history_get(&sh->history, hist_nav);
            if (hl) {
                int ol = len;
                shell_strncpy(buf, hl, SHELL_MAX_LINE); len = (int)shell_strlen(buf); cursor = len;
                line_refresh(sh->prompt, buf, len, cursor, ol);
            }
            continue;
        }

        /* Down: history next */
        if (key == KEY_DOWN) {
            if (hist_nav >= sh->history.count) continue;
            hist_nav++;
            int ol = len;
            if (hist_nav == sh->history.count) {
                for (int i = 0; i <= saved_len; i++) buf[i] = saved_line[i];
                len = saved_len;
            } else {
                const char *hl = history_get(&sh->history, hist_nav);
                if (hl) { shell_strncpy(buf, hl, SHELL_MAX_LINE); len = (int)shell_strlen(buf); }
            }
            cursor = len;
            line_refresh(sh->prompt, buf, len, cursor, ol);
            continue;
        }

        /* PgUp: jump to oldest history */
        if (key == KEY_PGUP) {
            int oldest = sh->history.count > SHELL_MAX_HISTORY ? sh->history.count - SHELL_MAX_HISTORY : 0;
            if (hist_nav == sh->history.count && len > 0) {
                for (int i = 0; i <= len; i++) saved_line[i] = buf[i]; saved_len = len;
            }
            hist_nav = oldest;
            const char *hl = history_get(&sh->history, hist_nav);
            if (hl) {
                int ol = len;
                shell_strncpy(buf, hl, SHELL_MAX_LINE); len = (int)shell_strlen(buf); cursor = len;
                line_refresh(sh->prompt, buf, len, cursor, ol);
            }
            continue;
        }

        /* PgDn: jump to current (newest) */
        if (key == KEY_PGDN) {
            hist_nav = sh->history.count;
            int ol = len;
            for (int i = 0; i <= saved_len; i++) buf[i] = saved_line[i];
            len = saved_len; cursor = len;
            line_refresh(sh->prompt, buf, len, cursor, ol);
            continue;
        }

        /* Printable character — insert at cursor */
        if (key >= 32 && key < 127 && len < max - 1) {
            for (int i = len; i > cursor; i--) buf[i] = buf[i - 1];
            buf[cursor] = (char)key;
            len++; buf[len] = '\0';
            for (int i = cursor; i < len; i++) { char ch[2] = {buf[i], 0}; kprintf("%s", ch); }
            cursor++;
            for (int i = len; i > cursor; i--) kprintf("\b");
        }
    }
}

/* ---- Parser: split line into argv ---- */

static void shell_parse_line(const char *line, int *argc, char *argv[])
{
    *argc = 0;
    const char *p = line;
    static char token_buf[SHELL_MAX_LINE];
    char *t = token_buf;

    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0') break;

        argv[*argc] = t;

        if (*p == '"') {
            p++;
            while (*p && *p != '"') *t++ = *p++;
            if (*p == '"') p++;
        } else {
            while (*p && *p != ' ' && *p != '\t') *t++ = *p++;
        }
        *t++ = '\0';
        (*argc)++;
        if (*argc >= SHELL_MAX_ARGS) break;
    }
}

/* =============================================================================
 * FILESYSTEM COMMANDS — wired to real TensorFS APIs
 * =============================================================================*/

static int cmd_ls(aishell_t *sh, int argc, char **argv)
{
    const char *path = argc >= 2 ? argv[1] : "/";
    tfs_inode_t entries[64];
    uint32_t count = 0;

    int r = tfs_readdir(path, entries, 64, &count);
    if (r < 0) {
        kprintf("ls: cannot access '%s': no such directory\n", path);
        return 1;
    }

    kprintf("\n  %-32s  %-10s  %s\n", "NAME", "SIZE", "TYPE");
    kprintf("  %-32s  %-10s  %s\n", "----", "----", "----");
    for (uint32_t i = 0; i < count; i++) {
        const char *type_str;
        switch (entries[i].type) {
            case TFS_FILE_DIR:       type_str = "dir";        break;
            case TFS_FILE_MODEL:     type_str = "model";      break;
            case TFS_FILE_WEIGHTS:   type_str = "weights";    break;
            case TFS_FILE_DATASET:   type_str = "dataset";    break;
            case TFS_FILE_CONFIG:    type_str = "config";     break;
            case TFS_FILE_TOKENIZER: type_str = "tokenizer";  break;
            case TFS_FILE_CHECKPOINT:type_str = "checkpoint"; break;
            case TFS_FILE_LOG:       type_str = "log";        break;
            default:                 type_str = "file";       break;
        }

        if (entries[i].size >= 1024*1024)
            kprintf("  %-32s  %6lu MB   %s\n", entries[i].name,
                    entries[i].size / (1024*1024), type_str);
        else if (entries[i].size >= 1024)
            kprintf("  %-32s  %6lu KB   %s\n", entries[i].name,
                    entries[i].size / 1024, type_str);
        else
            kprintf("  %-32s  %6lu B    %s\n", entries[i].name,
                    entries[i].size, type_str);
    }
    kprintf("\n  %d entries\n\n", count);
    return 0;
}

static int cmd_cat(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: cat <file>\n");
        return 1;
    }

    tfs_inode_t info;
    if (tfs_stat(argv[1], &info) < 0) {
        kprintf("cat: %s: no such file\n", argv[1]);
        return 1;
    }

    if (info.type == TFS_FILE_DIR) {
        kprintf("cat: %s: is a directory\n", argv[1]);
        return 1;
    }

    int fd = tfs_open(argv[1], 0);
    if (fd < 0) {
        kprintf("cat: %s: cannot open\n", argv[1]);
        return 1;
    }

    /* Read and print in chunks; limit output to 4KB for safety */
    static char buf[4096];
    uint64_t to_read = info.size < sizeof(buf) - 1 ? info.size : sizeof(buf) - 1;
    int n = tfs_read(fd, buf, to_read, 0);
    tfs_close(fd);

    if (n > 0) {
        buf[n] = '\0';
        kprintf("%s", buf);
        if (buf[n - 1] != '\n') kprintf("\n");
    }
    return 0;
}

static int cmd_mkdir(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: mkdir <path>\n");
        return 1;
    }
    int r = tfs_mkdir(argv[1]);
    if (r < 0) {
        kprintf("mkdir: cannot create '%s' (error %d)\n", argv[1], r);
        return 1;
    }
    kprintf("Created directory: %s\n", argv[1]);
    return 0;
}

static int cmd_touch(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: touch <file>\n");
        return 1;
    }
    int r = tfs_create(argv[1], TFS_FILE_REGULAR);
    if (r < 0) {
        kprintf("touch: cannot create '%s' (error %d)\n", argv[1], r);
        return 1;
    }
    kprintf("Created file: %s\n", argv[1]);
    return 0;
}

static int cmd_rm(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: rm <file>\n");
        return 1;
    }
    int r = tfs_unlink(argv[1]);
    if (r < 0) {
        kprintf("rm: cannot remove '%s' (error %d)\n", argv[1], r);
        return 1;
    }
    kprintf("Removed: %s\n", argv[1]);
    return 0;
}

static int cmd_write(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) {
        kprintf("Usage: write <file> <text...>\n");
        kprintf("  Writes text content into a file.\n");
        return 1;
    }

    /* Create if needed */
    tfs_create(argv[1], TFS_FILE_REGULAR);

    int fd = tfs_open(argv[1], 1 /* write */);
    if (fd < 0) {
        kprintf("write: cannot open '%s'\n", argv[1]);
        return 1;
    }

    /* Concatenate remaining args */
    static char content[2048];
    int pos = 0;
    for (int i = 2; i < argc && pos < 2040; i++) {
        if (i > 2 && pos < 2040) content[pos++] = ' ';
        for (const char *p = argv[i]; *p && pos < 2040; p++)
            content[pos++] = *p;
    }
    content[pos++] = '\n';
    content[pos] = '\0';

    tfs_write(fd, content, pos, 0);
    tfs_close(fd);

    kprintf("Wrote %d bytes to %s\n", pos, argv[1]);
    return 0;
}

static int cmd_stat(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: stat <path>\n");
        return 1;
    }
    tfs_inode_t info;
    if (tfs_stat(argv[1], &info) < 0) {
        kprintf("stat: %s: not found\n", argv[1]);
        return 1;
    }

    kprintf("\n  File: %s\n", info.name);
    kprintf("  Inode: %lu\n", info.inode_num);
    kprintf("  Type: %s\n",
            info.type == TFS_FILE_DIR ? "directory" :
            info.type == TFS_FILE_MODEL ? "model" :
            info.type == TFS_FILE_WEIGHTS ? "weights" :
            info.type == TFS_FILE_DATASET ? "dataset" :
            info.type == TFS_FILE_CHECKPOINT ? "checkpoint" : "regular");
    kprintf("  Size: %lu bytes", info.size);
    if (info.size >= 1024*1024)
        kprintf(" (%lu MB)", info.size / (1024*1024));
    kprintf("\n");
    kprintf("  Blocks: %lu (%lu bytes)\n", info.block_count,
            info.block_count * 4096);
    kprintf("  Checksum: 0x%08x\n", info.checksum);
    if (info.type == TFS_FILE_MODEL || info.type == TFS_FILE_WEIGHTS) {
        kprintf("  Model arch: %s\n", info.model_arch);
        kprintf("  Format: %s\n", info.format);
        kprintf("  Parameters: %lu\n", info.param_count);
    }
    kprintf("\n");
    return 0;
}

/* =============================================================================
 * HARDWARE INFO COMMANDS — wired to real kernel APIs
 * =============================================================================*/

static int cmd_cpu(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n=== CPU Information ===\n");
    kprintf("  Vendor:     %s\n", cpu_features.vendor);
    kprintf("  Cores:      %d (BSP APIC %d)\n", smp.cpu_count, smp.bsp_id);
    kprintf("  TSC:        %lu MHz\n", perf_tsc_mhz());

    kprintf("  ISA:       ");
    if (cpu_features.has_sse)    kprintf(" SSE");
    if (cpu_features.has_sse2)   kprintf(" SSE2");
    if (cpu_features.has_sse3)   kprintf(" SSE3");
    if (cpu_features.has_ssse3)  kprintf(" SSSE3");
    if (cpu_features.has_sse41)  kprintf(" SSE4.1");
    if (cpu_features.has_sse42)  kprintf(" SSE4.2");
    if (cpu_features.has_avx)    kprintf(" AVX");
    if (cpu_features.has_avx2)   kprintf(" AVX2");
    if (cpu_features.has_fma)    kprintf(" FMA");
    if (cpu_features.has_avx512f) kprintf(" AVX-512");
    if (cpu_features.has_aes)    kprintf(" AES-NI");
    if (cpu_features.has_popcnt) kprintf(" POPCNT");
    if (cpu_features.has_bmi1)   kprintf(" BMI1");
    if (cpu_features.has_bmi2)   kprintf(" BMI2");
    kprintf("\n");

    if (cpu_features.avx2_usable)
        kprintf("  GEMM:       AVX2+FMA 256-bit\n");
    else if (cpu_features.has_sse2)
        kprintf("  GEMM:       SSE2 128-bit\n");
    else
        kprintf("  GEMM:       scalar\n");

    kprintf("\n  Per-core status:\n");
    for (uint32_t i = 0; i < smp.cpu_count && i < MAX_CPUS; i++) {
        kprintf("    CPU %d: APIC %d, %s\n", i, smp.cpus[i].apic_id,
                smp.cpus[i].state == CPU_STATE_IDLE ? "IDLE" :
                smp.cpus[i].state == CPU_STATE_BUSY ? "BUSY" :
                smp.cpus[i].state == CPU_STATE_BOOTING ? "BOOTING" : "OFFLINE");
    }
    kprintf("\n");
    return 0;
}

static int cmd_disk(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n=== Disk Information ===\n");
    uint64_t cap = virtio_blk_capacity();
    if (cap == 0) {
        kprintf("  No block device available.\n\n");
        return 0;
    }
    virtio_blk_print_info();
    kprintf("\n");
    return 0;
}

static int cmd_smp(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n");
    smp_print_status();
    kprintf("\n");
    return 0;
}

static int cmd_net(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n=== Network ===\n");
    netstack_print_stats();
    if (!netstack_server_running()) {
        kprintf("\n  Tip: Run 'serve' to start the LLM API server.\n");
    }
    kprintf("\n");
    return 0;
}

/* =============================================================================
 * JIT COMPILER STATS
 * =============================================================================*/

static int cmd_jit(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n=== JIT Compiler ===\n");
    kprintf("  Backend:    x86_64 SSE2\n");
    kprintf("  Kernels:    %d compiled\n", jit_kernel_count());
    kprintf("  Code size:  %d bytes\n", jit_code_bytes());
    kprintf("  Pool:       1 MB static\n");
    kprintf("  Types:      matmul, relu, fused-matmul-relu, Q8_0-GEMV\n");

    if (argc >= 2 && shell_strcmp(argv[1], "test") == 0) {
        kprintf("\n  Running JIT self-test...\n");
        int pass = jit_selftest();
        kprintf("  Result: %s\n", pass ? "PASS" : "FAIL");
    }
    kprintf("\n");
    return 0;
}

/* =============================================================================
 * GIT COMMANDS — wired to real kernel git subsystem
 * =============================================================================*/

static git_repo_t shell_repo;
static int repo_initialized = 0;

static void ensure_repo(void)
{
    if (!repo_initialized) {
        if (git_repo_open("/", &shell_repo) < 0) {
            if (git_repo_init("/", &shell_repo) == 0) {
                repo_initialized = 1;
            }
        } else {
            repo_initialized = 1;
        }
    }
}

static int cmd_git(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: git <init|add|commit|log|status|branch|diff>\n");
        return 1;
    }

    ensure_repo();

    if (shell_strcmp(argv[1], "init") == 0) {
        int r = git_repo_init(argc >= 3 ? argv[2] : "/", &shell_repo);
        if (r == 0) {
            repo_initialized = 1;
            kprintf("Initialized empty TensorOS git repository\n");
        } else {
            kprintf("Failed to initialize repository (error %d)\n", r);
        }
        return r;
    }

    if (!repo_initialized) {
        kprintf("Not a git repository. Run 'git init' first.\n");
        return 1;
    }

    if (shell_strcmp(argv[1], "add") == 0) {
        if (argc < 3) {
            kprintf("Usage: git add <file>\n");
            return 1;
        }
        int r = git_add(&shell_repo, argv[2]);
        if (r == 0)
            kprintf("Added '%s' to staging area\n", argv[2]);
        else
            kprintf("Failed to add '%s' (error %d)\n", argv[2], r);
        return r;
    }

    if (shell_strcmp(argv[1], "commit") == 0) {
        const char *msg = "no message";
        if (argc >= 4 && shell_strcmp(argv[2], "-m") == 0)
            msg = argv[3];
        else if (argc >= 3)
            msg = argv[2];

        int r = git_commit(&shell_repo, msg);
        if (r == 0) {
            char hex[65];
            git_hash_to_hex(&shell_repo.head.target, hex);
            kprintf("[main %.7s] %s\n", hex, msg);
        } else {
            kprintf("Commit failed (error %d)\n", r);
        }
        return r;
    }

    if (shell_strcmp(argv[1], "log") == 0) {
        git_commit_t commits[16];
        uint32_t count = 0;
        int r = git_log(&shell_repo, commits, 16, &count);
        if (r < 0 || count == 0) {
            kprintf("No commits yet.\n");
            return 0;
        }
        kprintf("\n");
        for (uint32_t i = 0; i < count; i++) {
            char hex[65];
            git_hash_to_hex(&commits[i].header.hash, hex);
            kprintf("commit %.7s\n", hex);
            kprintf("Author: %s\n", commits[i].author);
            kprintf("\n    %s\n\n", commits[i].message);
        }
        return 0;
    }

    if (shell_strcmp(argv[1], "status") == 0) {
        kprintf("On branch main\n");
        kprintf("Objects: %lu, Refs: %u\n",
                shell_repo.object_count, shell_repo.ref_count);
        return 0;
    }

    if (shell_strcmp(argv[1], "branch") == 0) {
        if (argc >= 3) {
            int r = git_branch_create(&shell_repo, argv[2]);
            if (r == 0)
                kprintf("Created branch '%s'\n", argv[2]);
            else
                kprintf("Failed to create branch (error %d)\n", r);
            return r;
        }
        /* List branches */
        git_ref_t refs[32];
        uint32_t count = 0;
        git_ref_list(&shell_repo, refs, 32, &count);
        for (uint32_t i = 0; i < count; i++) {
            char hex[65];
            git_hash_to_hex(&refs[i].target, hex);
            kprintf("  %s%.7s %s\n",
                    shell_strcmp(refs[i].name, shell_repo.head.symref) == 0 ? "* " : "  ",
                    hex, refs[i].name);
        }
        if (count == 0) kprintf("  * main (no commits)\n");
        return 0;
    }

    if (shell_strcmp(argv[1], "diff") == 0) {
        if (argc < 4) {
            kprintf("Usage: git diff <hash-a> <hash-b>\n");
            return 1;
        }
        kprintf("[GIT] Diff requires two tree hashes.\n");
        return 0;
    }

    kprintf("Unknown git subcommand: %s\n", argv[1]);
    return 1;
}

/* =============================================================================
 * MODEL MANAGEMENT — wired to real MEU/scheduler APIs
 * =============================================================================*/

static int cmd_model(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: model <list|create|load|info|kill> [args...]\n");
        return 1;
    }

    if (shell_strcmp(argv[1], "list") == 0) {
        kprintf("\n  ID   STATE      PRIORITY   NAME                 PARAMS       MEM\n");
        kprintf("  ---  ---------  ---------  -------------------  -----------  -----------\n");
        for (uint32_t i = 0; i < kstate.meu_count; i++) {
            model_exec_unit_t *meu = &kstate.meus[i];
            const char *state_str =
                meu->state == MEU_STATE_RUNNING   ? "RUNNING" :
                meu->state == MEU_STATE_READY     ? "READY"   :
                meu->state == MEU_STATE_LOADING   ? "LOADING" :
                meu->state == MEU_STATE_WAITING   ? "WAITING" :
                meu->state == MEU_STATE_SUSPENDED ? "SUSPEND" :
                meu->state == MEU_STATE_COMPLETED ? "DONE"    :
                meu->state == MEU_STATE_ERROR     ? "ERROR"   : "CREATED";
            const char *pri_str =
                meu->priority == MEU_PRIO_REALTIME ? "REALTIME" :
                meu->priority == MEU_PRIO_HIGH     ? "HIGH"     :
                meu->priority == MEU_PRIO_NORMAL   ? "NORMAL"   :
                meu->priority == MEU_PRIO_LOW      ? "LOW"      : "BG";
            kprintf("  %3u  %-9s  %-9s  %-19s  %11lu  %lu MB\n",
                    (uint32_t)meu->meu_id, state_str, pri_str,
                    meu->name, meu->param_count,
                    meu->mem_used / (1024*1024));
        }
        if (llm_is_loaded()) {
            kprintf("  ---  RUNNING    HIGH       %-19s  %11s  ---\n",
                    llm_model_name(), "GGUF Q8_0");
        }
        kprintf("\n  Total MEUs: %d", kstate.meu_count);
        if (llm_is_loaded()) kprintf(" (+1 LLM)");
        kprintf("\n\n");
        return 0;
    }

    if (shell_strcmp(argv[1], "create") == 0) {
        if (argc < 3) {
            kprintf("Usage: model create <name> [priority]\n");
            kprintf("  priority: realtime|high|normal|low|bg\n");
            return 1;
        }
        meu_priority_t pri = MEU_PRIO_NORMAL;
        if (argc >= 4) {
            if (shell_strcmp(argv[3], "realtime") == 0) pri = MEU_PRIO_REALTIME;
            else if (shell_strcmp(argv[3], "high") == 0) pri = MEU_PRIO_HIGH;
            else if (shell_strcmp(argv[3], "low") == 0) pri = MEU_PRIO_LOW;
            else if (shell_strcmp(argv[3], "bg") == 0) pri = MEU_PRIO_BACKGROUND;
        }
        model_exec_unit_t *meu = meu_create(argv[2], MEU_TYPE_INFERENCE, pri);
        if (meu) {
            kprintf("Created MEU #%lu '%s'\n", meu->meu_id, argv[2]);
            tensor_sched_enqueue(meu);
        } else {
            kprintf("Failed to create MEU (max reached?)\n");
        }
        return meu ? 0 : 1;
    }

    if (shell_strcmp(argv[1], "load") == 0) {
        if (argc < 3) {
            kprintf("Usage: model load <name>\n");
            return 1;
        }
        kprintf("[MODEL] Creating MEU for '%s'...\n", argv[2]);
        model_exec_unit_t *meu = meu_create(argv[2], MEU_TYPE_INFERENCE,
                                              MEU_PRIO_NORMAL);
        if (meu) {
            tensor_sched_enqueue(meu);
            kprintf("[MODEL] MEU #%lu created and enqueued.\n", meu->meu_id);
        } else {
            kprintf("[MODEL] Failed to create MEU.\n");
        }
        return meu ? 0 : 1;
    }

    if (shell_strcmp(argv[1], "info") == 0) {
        if (argc < 3) {
            kprintf("Usage: model info <id>\n");
            return 1;
        }
        int id = shell_atoi(argv[2]);
        if (id < 0 || (uint32_t)id >= kstate.meu_count) {
            kprintf("Invalid MEU ID: %d\n", id);
            return 1;
        }
        model_exec_unit_t *meu = &kstate.meus[id];
        kprintf("\n=== MEU #%lu ===\n", meu->meu_id);
        kprintf("  Name:       %s\n", meu->name);
        kprintf("  State:      %d\n", meu->state);
        kprintf("  Type:       %s\n",
                meu->type == MEU_TYPE_INFERENCE ? "inference" :
                meu->type == MEU_TYPE_TRAINING  ? "training"  :
                meu->type == MEU_TYPE_FINETUNE  ? "finetune"  : "other");
        kprintf("  Parameters: %lu\n", meu->param_count);
        kprintf("  Mem budget: %lu MB (used %lu MB)\n",
                meu->mem_budget / (1024*1024), meu->mem_used / (1024*1024));
        kprintf("  VRAM:       %lu MB (used %lu MB)\n",
                meu->vram_budget / (1024*1024), meu->vram_used / (1024*1024));
        kprintf("  Tensor ops: %lu\n", meu->tensor_ops);
        kprintf("  FLOPS:      %lu\n", meu->flops);
        kprintf("  Inferences: %lu\n", meu->inferences);
        kprintf("  CPU ticks:  %lu\n", meu->cpu_ticks);
        kprintf("  GPU ID:     %u\n", meu->gpu_id);
        kprintf("\n");
        return 0;
    }

    if (shell_strcmp(argv[1], "kill") == 0) {
        if (argc < 3) {
            kprintf("Usage: model kill <id>\n");
            return 1;
        }
        int id = shell_atoi(argv[2]);
        if (id < 0 || (uint32_t)id >= kstate.meu_count) {
            kprintf("Invalid MEU ID: %d\n", id);
            return 1;
        }
        model_exec_unit_t *meu = &kstate.meus[id];
        kprintf("Killing MEU #%lu '%s'...\n", meu->meu_id, meu->name);
        tensor_sched_dequeue(meu);
        meu_destroy(meu);
        kprintf("MEU destroyed.\n");
        return 0;
    }

    kprintf("Unknown model subcommand: %s\n", argv[1]);
    kprintf("  Use: list, create, load, info, kill\n");
    return 1;
}

/* =============================================================================
 * MEMORY — enhanced with full mm_stats
 * =============================================================================*/

static int cmd_mem(aishell_t *sh, int argc, char **argv)
{
    mm_stats_t stats;
    tensor_mm_get_stats(&stats);

    kprintf("\n=== Memory ===\n");
    kprintf("  Physical:     %lu MB total, %lu MB free\n",
            stats.total_phys / (1024*1024), stats.free_phys / (1024*1024));
    kprintf("  Tensor heap:  %lu MB (used %lu MB)\n",
            stats.tensor_heap_size / (1024*1024),
            stats.tensor_heap_used / (1024*1024));
    kprintf("  Model cache:  %lu MB (used %lu MB)\n",
            stats.model_cache_size / (1024*1024),
            stats.model_cache_used / (1024*1024));
    if (stats.gpu_mem_total > 0) {
        kprintf("  GPU memory:   %lu MB (used %lu MB)\n",
                stats.gpu_mem_total / (1024*1024),
                stats.gpu_mem_used / (1024*1024));
    }
    kprintf("  Allocations:  %lu alloc, %lu free\n",
            stats.alloc_count, stats.free_count);
    if (stats.huge_pages_used > 0)
        kprintf("  Huge pages:   %lu in use\n", stats.huge_pages_used);
    if (stats.page_faults > 0)
        kprintf("  Page faults:  %lu\n", stats.page_faults);

    if (argc >= 2 && shell_strcmp(argv[1], "defrag") == 0) {
        kprintf("\n  Running defragmentation...\n");
        tensor_mm_defrag();
        kprintf("  Done.\n");
    }
    kprintf("\n");
    return 0;
}

/* =============================================================================
 * SCHEDULER INFO
 * =============================================================================*/

static int cmd_sched(aishell_t *sh, int argc, char **argv)
{
    uint64_t ops = 0, dispatches = 0, avg_lat = 0;
    tensor_sched_get_stats(&ops, &dispatches, &avg_lat);

    kprintf("\n=== Tensor Scheduler ===\n");
    kprintf("  Policy:       %s\n",
            g_scheduler.policy == 0 ? "THROUGHPUT" :
            g_scheduler.policy == 1 ? "LATENCY"    :
            g_scheduler.policy == 2 ? "EFFICIENCY" : "FAIR");
    kprintf("  Dispatches:   %lu\n", dispatches);
    kprintf("  Preemptions:  %lu\n", g_scheduler.total_preemptions);
    kprintf("  Migrations:   %lu\n", g_scheduler.total_migrations);
    kprintf("  Tensor ops:   %lu\n", ops);
    kprintf("  Avg latency:  %lu us\n", avg_lat);
    kprintf("  GPUs:         %u\n", g_scheduler.gpu_count);
    kprintf("  TPUs:         %u\n", g_scheduler.tpu_count);
    kprintf("  Coalesce:     %s",
            g_scheduler.coalesce_window_ms > 0 ? "ON" : "OFF");
    if (g_scheduler.coalesce_window_ms > 0)
        kprintf(" (%u ms window, %u pending)", g_scheduler.coalesce_window_ms,
                g_scheduler.pending_batch_count);
    kprintf("\n");

    /* Per-priority queue depths */
    kprintf("\n  Queue depths:\n");
    const char *pri_names[] = {"REALTIME", "HIGH", "NORMAL", "LOW", "BG", "IDLE"};
    for (int i = 0; i < 6; i++) {
        if (g_scheduler.queues[i].count > 0)
            kprintf("    %-10s: %u MEUs\n", pri_names[i],
                    g_scheduler.queues[i].count);
    }

    if (argc >= 2 && shell_strcmp(argv[1], "balance") == 0) {
        kprintf("\n  Rebalancing devices...\n");
        tensor_sched_balance_devices();
        kprintf("  Done.\n");
    }
    if (argc >= 2 && shell_strcmp(argv[1], "flush") == 0) {
        kprintf("\n  Flushing coalesced batches...\n");
        tensor_sched_coalesce_flush();
        kprintf("  Done.\n");
    }
    kprintf("\n");
    return 0;
}

/* =============================================================================
 * INTERACTIVE LLM PROMPT
 * =============================================================================*/

static int cmd_llm(aishell_t *sh, int argc, char **argv)
{
    if (!llm_is_loaded()) {
        kprintf("[LLM] No model loaded. Attach a GGUF model disk to use this.\n");
        return 1;
    }

    if (argc < 2) {
        kprintf("[LLM] Model: %s\n", llm_model_name());
        kprintf("Usage: llm <your prompt text>\n");
        kprintf("  e.g. llm What is 2+2?\n");
        kprintf("  e.g. llm Explain neural networks briefly\n");
        kprintf("  TIP: 'ai' is a shorthand alias for 'llm'\n");
        return 0;
    }

    static char prompt[1024];
    int pos = 0;
    for (int i = 1; i < argc; i++) {
        if (i > 1 && pos < 1020) prompt[pos++] = ' ';
        for (const char *p = argv[i]; *p && pos < 1020; p++)
            prompt[pos++] = *p;
    }
    prompt[pos] = '\0';

    kprintf("[LLM] Model: %s\n", llm_model_name());
    kprintf("[LLM] Prompt: %s\n", prompt);
    kprintf("[LLM] Generating...\n");

    static char response[2048];
    uint64_t t0 = rdtsc_fenced();
    int n_gen = llm_prompt(prompt, response, sizeof(response));
    uint64_t t1 = rdtsc_fenced();
    uint64_t ms = perf_cycles_to_us(t1 - t0) / 1000;

    if (n_gen < 0) {
        kprintf("[LLM] Error generating response.\n");
        return 1;
    }

    kprintf("\n%s\n\n", response);
    kprintf("[LLM] %d tokens in %lu ms", n_gen, ms);
    if (n_gen > 0 && ms > 0) {
        uint64_t tps = ((uint64_t)n_gen * 1000) / ms;
        kprintf(" (%lu ms/tok, ~%lu tok/s)", ms / (uint64_t)n_gen, tps);
    }
    kprintf("\n");
    return 0;
}

static int cmd_reset(aishell_t *sh, int argc, char **argv)
{
    extern void llm_reset_cache(void);
    llm_reset_cache();
    kprintf("[LLM] Conversation reset. KV cache cleared.\n");
    return 0;
}

/* =============================================================================
 * UTILITY COMMANDS
 * =============================================================================*/

static int cmd_history(aishell_t *sh, int argc, char **argv)
{
    shell_history_t *h = &sh->history;
    int total = h->count < SHELL_MAX_HISTORY ? h->count : SHELL_MAX_HISTORY;
    int start = h->count < SHELL_MAX_HISTORY ? 0 : h->count - SHELL_MAX_HISTORY;

    kprintf("\n");
    for (int i = 0; i < total; i++) {
        int idx = (start + i) % SHELL_MAX_HISTORY;
        kprintf("  %3d  %s\n", start + i + 1, h->lines[idx]);
    }
    kprintf("\n");
    return 0;
}

static int cmd_uptime(aishell_t *sh, int argc, char **argv)
{
    extern uint64_t watchdog_uptime_ms(void);
    uint64_t ms = watchdog_uptime_ms();
    uint64_t sec = ms / 1000;
    uint64_t min = sec / 60;
    uint64_t hr  = min / 60;
    kprintf("up %lu:%02lu:%02lu (%lu ms)\n", hr, min % 60, sec % 60, ms);
    return 0;
}

static int cmd_echo(aishell_t *sh, int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if (i > 1) kprintf(" ");
        kprintf("%s", argv[i]);
    }
    kprintf("\n");
    return 0;
}

static int cmd_clear(aishell_t *sh, int argc, char **argv)
{
    vga_init();
    return 0;
}

static int cmd_uname(aishell_t *sh, int argc, char **argv)
{
    int all = (argc >= 2 && shell_strcmp(argv[1], "-a") == 0);
#if defined(__aarch64__)
    kprintf("TensorOS v0.1.0 \"Neuron\" aarch64 NEON");
#else
    kprintf("TensorOS v0.1.0 \"Neuron\" x86_64");
    if (cpu_features.avx2_usable) kprintf(" AVX2+FMA");
    else kprintf(" SSE2");
#endif
    if (all) {
        kprintf(" %d CPU(s) %lu MHz %lu MB RAM",
                smp.cpu_count, perf_tsc_mhz(),
                kstate.memory_total_bytes / (1024*1024));
    }
    kprintf("\n");
    return 0;
}

static int cmd_reboot(aishell_t *sh, int argc, char **argv)
{
    kprintf("[REBOOT] Resetting system...\n");
#ifndef __aarch64__
    __asm__ volatile ("cli");
    struct { uint16_t limit; uint64_t base; } __attribute__((packed)) null_idtr = {0, 0};
    __asm__ volatile ("lidt %0" : : "m"(null_idtr));
    __asm__ volatile ("int $0x03");
#else
    __asm__ volatile ("mov x0, #0x84000009; hvc #0");
#endif
    while(1) __asm__ volatile ("hlt");
    return 0;
}

/* Hex dump memory at address */
static int cmd_hexdump(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: hexdump <address> [length]\n");
        kprintf("  address: hex (e.g. 0x200000) or decimal\n");
        kprintf("  length:  bytes to dump (default 256, max 4096)\n");
        return 1;
    }

    uint64_t addr = shell_atou64(argv[1]);
    int len = argc >= 3 ? shell_atoi(argv[2]) : 256;
    if (len <= 0) len = 256;
    if (len > 4096) len = 4096;

    const uint8_t *p = (const uint8_t *)addr;
    kprintf("\n");
    for (int off = 0; off < len; off += 16) {
        kprintf("  %016lx: ", addr + off);
        int row = 16;
        if (off + row > len) row = len - off;
        for (int j = 0; j < row; j++)
            kprintf("%02x ", p[off + j]);
        for (int j = row; j < 16; j++)
            kprintf("   ");
        kprintf(" ");
        for (int j = 0; j < row; j++) {
            uint8_t c = p[off + j];
            kprintf("%c", (c >= 0x20 && c < 0x7F) ? c : '.');
        }
        kprintf("\n");
    }
    kprintf("\n");
    return 0;
}

/* Simple integer calculator */
static int cmd_calc(aishell_t *sh, int argc, char **argv)
{
    if (argc < 4) {
        kprintf("Usage: calc <a> <op> <b>\n");
        kprintf("  ops: + - * / %% & | ^ << >>\n");
        kprintf("  e.g.  calc 42 + 13\n");
        kprintf("  e.g.  calc 0xFF & 0x0F\n");
        return 1;
    }

    int64_t a = (int64_t)shell_atou64(argv[1]);
    int64_t b = (int64_t)shell_atou64(argv[3]);
    const char *op = argv[2];
    int64_t result = 0;
    int valid = 1;

    if (shell_strcmp(op, "+") == 0)       result = a + b;
    else if (shell_strcmp(op, "-") == 0)  result = a - b;
    else if (shell_strcmp(op, "*") == 0)  result = a * b;
    else if (shell_strcmp(op, "/") == 0) {
        if (b == 0) { kprintf("Division by zero\n"); return 1; }
        result = a / b;
    }
    else if (shell_strcmp(op, "%") == 0) {
        if (b == 0) { kprintf("Division by zero\n"); return 1; }
        result = a % b;
    }
    else if (shell_strcmp(op, "&") == 0)  result = a & b;
    else if (shell_strcmp(op, "|") == 0)  result = a | b;
    else if (shell_strcmp(op, "^") == 0)  result = a ^ b;
    else if (shell_strcmp(op, "<<") == 0) result = a << b;
    else if (shell_strcmp(op, ">>") == 0) result = a >> b;
    else { kprintf("Unknown operator: %s\n", op); valid = 0; }

    if (valid)
        kprintf("= %ld (0x%lx)\n", result, (uint64_t)result);
    return valid ? 0 : 1;
}

/* Time a command's execution */
static int cmd_time(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: time <command> [args...]\n");
        return 1;
    }

    /* Rebuild a command line from argv[1..] */
    static char subcmd[SHELL_MAX_LINE];
    int pos = 0;
    for (int i = 1; i < argc && pos < SHELL_MAX_LINE - 2; i++) {
        if (i > 1) subcmd[pos++] = ' ';
        for (const char *p = argv[i]; *p && pos < SHELL_MAX_LINE - 2; p++)
            subcmd[pos++] = *p;
    }
    subcmd[pos] = '\0';

    int sub_argc;
    char *sub_argv[SHELL_MAX_ARGS];
    shell_parse_line(subcmd, &sub_argc, sub_argv);

    uint64_t t0 = rdtsc_fenced();
    int r = shell_exec_builtin(sh, sub_argc, sub_argv);
    uint64_t t1 = rdtsc_fenced();

    uint64_t us = perf_cycles_to_us(t1 - t0);
    if (us >= 1000000)
        kprintf("\nreal\t%lu.%03lu s\n", us / 1000000, (us / 1000) % 1000);
    else if (us >= 1000)
        kprintf("\nreal\t%lu.%03lu ms\n", us / 1000, us % 1000);
    else
        kprintf("\nreal\t%lu us\n", us);

    return r;
}

/* Process list */
static int cmd_ps(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n  PID  STATE      TYPE       NAME\n");
    kprintf("  ---  ---------  ---------  --------------------\n");
    kprintf("  0    RUNNING    kernel     TensorOS kernel\n");
    kprintf("  1    RUNNING    shell      aishell\n");
    for (uint32_t i = 0; i < kstate.meu_count; i++) {
        model_exec_unit_t *meu = &kstate.meus[i];
        kprintf("  %3u  %-9s  %-9s  %s\n",
                i + 2,
                meu->state == MEU_STATE_RUNNING ? "RUNNING" :
                meu->state == MEU_STATE_READY   ? "READY"   :
                meu->state == MEU_STATE_LOADING ? "LOADING" :
                meu->state == MEU_STATE_WAITING ? "WAITING" :
                meu->state == MEU_STATE_ERROR   ? "ERROR"   : "STOPPED",
                meu->type == MEU_TYPE_INFERENCE ? "inference" :
                meu->type == MEU_TYPE_TRAINING  ? "training"  :
                meu->type == MEU_TYPE_FINETUNE  ? "finetune"  : "system",
                meu->name);
    }
    if (llm_is_loaded()) {
        kprintf("  %3u  RUNNING    llm        %s\n",
                kstate.meu_count + 2, llm_model_name());
    }
    kprintf("\n");
    return 0;
}

/* PCI device listing */
static int cmd_lspci(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n=== PCI Devices ===\n");
    extern void pci_enumerate(void);
    pci_enumerate();
    kprintf("\n");
    return 0;
}

/* System monitor */
static int cmd_monitor(aishell_t *sh, int argc, char **argv)
{
    extern uint64_t watchdog_uptime_ms(void);
    mm_stats_t stats;
    tensor_mm_get_stats(&stats);
    uint64_t sched_ops = 0, sched_disp = 0, sched_lat = 0;
    tensor_sched_get_stats(&sched_ops, &sched_disp, &sched_lat);

    kprintf("\n=== TensorOS System Monitor ===\n");
    kprintf("  Uptime:        %lu ms\n", watchdog_uptime_ms());
    kprintf("  CPUs:          %d (%s)\n", smp.cpu_count, cpu_features.vendor);
    kprintf("  TSC:           %lu MHz\n", perf_tsc_mhz());
    kprintf("  RAM:           %lu MB total, %lu MB free\n",
            stats.total_phys / (1024*1024), stats.free_phys / (1024*1024));
    kprintf("  Heap:          %lu / %lu MB\n",
            stats.tensor_heap_used / (1024*1024),
            stats.tensor_heap_size / (1024*1024));
    kprintf("  MEUs:          %d running\n", kstate.meu_count);
    kprintf("  Tensor ops:    %lu\n", sched_ops);
    kprintf("  Dispatches:    %lu (avg %lu us)\n", sched_disp, sched_lat);
    kprintf("  JIT kernels:   %d (%d bytes)\n", jit_kernel_count(), jit_code_bytes());
    kprintf("  Allocs:        %lu alloc, %lu free\n",
            stats.alloc_count, stats.free_count);
    if (llm_is_loaded())
        kprintf("  LLM:           %s\n", llm_model_name());
    kprintf("\n");
    return 0;
}

/* Status overview */
static int cmd_status(aishell_t *sh, int argc, char **argv)
{
    extern uint64_t tensor_mm_free_bytes(void);

    kprintf("\n=== TensorOS Status ===\n");
    kprintf("  Version:    v0.1.0 \"Neuron\"\n");
    kprintf("  Phase:      %s\n",
            kstate.phase == 0 ? "BOOT" :
            kstate.phase == 1 ? "INIT" :
            kstate.phase == 2 ? "RUNNING" : "PANIC");
    kprintf("  CPUs:       %d (%s)\n", kstate.cpu_count, cpu_features.vendor);
    kprintf("  GPUs:       %d\n", kstate.gpu_count);
    kprintf("  TSC:        %lu MHz\n", perf_tsc_mhz());
    kprintf("  Memory:     %lu MB free / %lu MB total\n",
            tensor_mm_free_bytes() / (1024*1024),
            kstate.memory_total_bytes / (1024*1024));
    kprintf("  MEUs:       %d running\n", kstate.meu_count);
    kprintf("  Tensor ops: %lu total\n", kstate.tensor_ops_total);
    kprintf("  JIT:        %d kernels (%d B code)\n",
            jit_kernel_count(), jit_code_bytes());
    if (cpu_features.avx2_usable)
        kprintf("  SIMD:       AVX2+FMA (256-bit)\n");
    else
        kprintf("  SIMD:       SSE2 (128-bit)\n");
    if (llm_is_loaded())
        kprintf("  LLM:        %s\n", llm_model_name());
    kprintf("  Shell cmds: %u executed\n", sh->commands_executed);
    kprintf("\n");
    return 0;
}

/* Comprehensive sysinfo */
static int cmd_sysinfo(aishell_t *sh, int argc, char **argv)
{
    extern uint64_t watchdog_uptime_ms(void);

    mm_stats_t stats;
    tensor_mm_get_stats(&stats);
    uint64_t ms = watchdog_uptime_ms();

    kprintf("\n=== TensorOS System Information ===\n\n");
#if defined(__aarch64__)
    kprintf("  OS:         TensorOS v0.1.0 \"Neuron\" (aarch64)\n");
    kprintf("  SIMD:       NEON 128-bit\n");
#else
    kprintf("  OS:         TensorOS v0.1.0 \"Neuron\" (x86_64)\n");
    if (cpu_features.avx2_usable)
        kprintf("  SIMD:       AVX2+FMA 256-bit\n");
    else
        kprintf("  SIMD:       SSE2 128-bit\n");
#endif
    kprintf("  CPU:        %s, %d cores @ %lu MHz\n",
            cpu_features.vendor, smp.cpu_count, perf_tsc_mhz());
    kprintf("  RAM:        %lu MB total, %lu MB free\n",
            stats.total_phys / (1024*1024), stats.free_phys / (1024*1024));
    kprintf("  Heap:       %lu MB / %lu MB\n",
            stats.tensor_heap_used / (1024*1024),
            stats.tensor_heap_size / (1024*1024));
    kprintf("  Cache:      %lu MB / %lu MB\n",
            stats.model_cache_used / (1024*1024),
            stats.model_cache_size / (1024*1024));

    uint64_t cap = virtio_blk_capacity();
    if (cap > 0)
        kprintf("  Disk:       %lu MB (virtio-blk)\n", cap / (1024*1024));
    else
        kprintf("  Disk:       none\n");

    kprintf("  Uptime:     %lu.%03lu s\n", ms / 1000, ms % 1000);
    kprintf("  MEUs:       %d loaded\n", kstate.meu_count);
    kprintf("  JIT:        %d kernels, %d B compiled\n",
            jit_kernel_count(), jit_code_bytes());
    kprintf("  Tensor ops: %lu\n", kstate.tensor_ops_total);
    if (llm_is_loaded())
        kprintf("  LLM:        %s (loaded)\n", llm_model_name());
    else
        kprintf("  LLM:        (none)\n");
    kprintf("\n");
    return 0;
}

/* =============================================================================
 * TENSOR OPERATIONS — wired to real tensor CPU math
 * =============================================================================*/

static int cmd_tensor(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: tensor <selftest|matmul|alloc|free> [args...]\n");
        return 1;
    }

    if (shell_strcmp(argv[1], "selftest") == 0) {
        extern int tensor_cpu_selftest(void);
        kprintf("[TENSOR] Running CPU tensor math self-test...\n");
        int pass = tensor_cpu_selftest();
        kprintf("[TENSOR] Result: %s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }

    if (shell_strcmp(argv[1], "matmul") == 0) {
        int N = argc >= 3 ? shell_atoi(argv[2]) : 64;
        if (N <= 0 || N > 512) N = 64;
        kprintf("[TENSOR] Benchmarking %dx%d matmul...\n", N, N);

        float *A = (float *)tensor_alloc(N * N * sizeof(float));
        float *B = (float *)tensor_alloc(N * N * sizeof(float));
        float *C = (float *)tensor_alloc(N * N * sizeof(float));
        if (!A || !B || !C) {
            kprintf("[TENSOR] Out of memory\n");
            if (A) tensor_free(A);
            if (B) tensor_free(B);
            if (C) tensor_free(C);
            return 1;
        }

        for (int i = 0; i < N * N; i++) { A[i] = 1.0f; B[i] = 1.0f; }

        extern void tensor_cpu_matmul(float *, const float *, const float *, int, int, int);
        uint64_t t0 = rdtsc_fenced();
        tensor_cpu_matmul(C, A, B, N, N, N);
        uint64_t t1 = rdtsc_fenced();
        uint64_t us = perf_cycles_to_us(t1 - t0);

        uint64_t flop = 2ULL * N * N * N;
        kprintf("[TENSOR] %dx%d: %lu us, C[0]=%d (expect %d)\n",
                N, N, us, (int)C[0], N);
        if (us > 0)
            kprintf("[TENSOR] ~%lu MFLOPS\n", flop / us);

        tensor_free(A);
        tensor_free(B);
        tensor_free(C);
        return 0;
    }

    if (shell_strcmp(argv[1], "alloc") == 0) {
        int bytes = argc >= 3 ? shell_atoi(argv[2]) : 1024;
        if (bytes <= 0) bytes = 1024;
        void *p = tensor_alloc((uint64_t)bytes);
        if (p)
            kprintf("Allocated %d bytes at 0x%lx\n", bytes, (uint64_t)p);
        else
            kprintf("Allocation failed\n");
        return p ? 0 : 1;
    }

    if (shell_strcmp(argv[1], "free") == 0) {
        if (argc < 3) {
            kprintf("Usage: tensor free <address>\n");
            return 1;
        }
        uint64_t addr = shell_atou64(argv[2]);
        tensor_free((void *)addr);
        kprintf("Freed 0x%lx\n", addr);
        return 0;
    }

    kprintf("Unknown tensor subcommand: %s\n", argv[1]);
    return 1;
}

/* =============================================================================
 * INFERENCE / TRAINING / DEPLOY — wired to real APIs
 * =============================================================================*/

static int cmd_infer(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: infer <prompt text...>\n");
        kprintf("  Runs LLM inference (shorthand for 'llm <prompt>').\n");
        return 1;
    }
    return cmd_llm(sh, argc, argv);
}

static int cmd_train(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: train <demo|xor>\n");
        kprintf("  demo    Run backprop+Adam training demos\n");
        kprintf("  xor     Tiny XOR network training\n");
        return 1;
    }
    extern void nn_train_demos(void);
    if (shell_strcmp(argv[1], "demo") == 0 || shell_strcmp(argv[1], "xor") == 0) {
        kprintf("[TRAIN] Running training demos...\n\n");
        nn_train_demos();
        return 0;
    }
    kprintf("Unknown training target: %s\n", argv[1]);
    return 1;
}

static int cmd_deploy(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: deploy <model-name> [port]\n");
        return 1;
    }
    int port = argc >= 3 ? shell_atoi(argv[2]) : 8080;
    if (port <= 0 || port > 65535) port = 8080;

    kprintf("[DEPLOY] Creating MEU for '%s'...\n", argv[1]);
    model_exec_unit_t *meu = meu_create(argv[1], MEU_TYPE_INFERENCE,
                                          MEU_PRIO_HIGH);
    if (!meu) {
        kprintf("[DEPLOY] Failed to create MEU.\n");
        return 1;
    }
    tensor_sched_enqueue(meu);
    kprintf("[DEPLOY] MEU #%lu enqueued.\n", meu->meu_id);
    kprintf("[DEPLOY] Starting HTTP endpoint on port %d...\n", port);
    netstack_start_http_server();
    kprintf("[DEPLOY] Model '%s' deployed. POST /infer -> port %d\n",
            argv[1], port);
    return 0;
}

/* =============================================================================
 * SERVE — Start OpenAI-compatible HTTP API server
 *
 * This is the primary way to expose the LLM over the network.
 * Any device can connect using curl, Python, or the OpenAI SDK.
 * =============================================================================*/

static int cmd_serve(aishell_t *sh, int argc, char **argv)
{
    (void)sh;

    if (netstack_server_running()) {
        const net_config_t *cfg = netstack_get_config();
        kprintf("Server already running at http://%u.%u.%u.%u:%u\n",
                cfg->ip[0], cfg->ip[1], cfg->ip[2], cfg->ip[3],
                cfg->http_port);
        kprintf("Use 'net' to see stats.\n");
        return 0;
    }

    if (!llm_is_loaded()) {
        kprintf("Warning: No LLM model loaded. API will return 503 until a model is loaded.\n");
        kprintf("Load a model first with: llm load\n\n");
    }

    /* Start the server */
    netstack_start_http_server();
    return 0;
}

/* =============================================================================
 * PACKAGE MANAGER — wired to real APIs
 * =============================================================================*/

static int cmd_pkg(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: pkg <install|search|list|remove|info|update|verify> [args]\n");
        return 1;
    }

    if (shell_strcmp(argv[1], "install") == 0) {
        if (argc < 3) { kprintf("Usage: pkg install <name> [version]\n"); return 1; }
        const char *ver = argc >= 4 ? argv[3] : "latest";
        kprintf("[PKG] Installing '%s' (%s)...\n", argv[2], ver);
        int r = modelpkg_install(argv[2], ver);
        kprintf("[PKG] %s\n", r == 0 ? "Installed successfully." : "Install failed.");
        return r;
    }

    if (shell_strcmp(argv[1], "search") == 0) {
        if (argc < 3) { kprintf("Usage: pkg search <query>\n"); return 1; }
        model_manifest_t results[16];
        uint32_t count = 0;
        modelpkg_search(argv[2], results, 16, &count);
        if (count == 0) {
            kprintf("No packages matching '%s'.\n", argv[2]);
        } else {
            kprintf("\n  %-30s  %-10s  %s\n", "NAME", "VERSION", "SIZE");
            kprintf("  %-30s  %-10s  %s\n", "----", "-------", "----");
            for (uint32_t i = 0; i < count; i++)
                kprintf("  %-30s  %-10s  %lu MB\n",
                        results[i].name, results[i].version,
                        results[i].total_size / (1024*1024));
            kprintf("\n  %d result(s)\n", count);
        }
        kprintf("\n");
        return 0;
    }

    if (shell_strcmp(argv[1], "list") == 0) {
        installed_model_t models[32];
        uint32_t count = 0;
        modelpkg_list_installed(models, 32, &count);
        if (count == 0) {
            kprintf("No models installed.\n");
        } else {
            kprintf("\n  %-30s  %-10s  %s\n", "NAME", "VERSION", "PATH");
            kprintf("  %-30s  %-10s  %s\n", "----", "-------", "----");
            for (uint32_t i = 0; i < count; i++)
                kprintf("  %-30s  %-10s  %s\n",
                        models[i].manifest.name, models[i].manifest.version,
                        models[i].install_path);
            kprintf("\n  %d installed\n", count);
        }
        kprintf("\n");
        return 0;
    }

    if (shell_strcmp(argv[1], "remove") == 0 || shell_strcmp(argv[1], "uninstall") == 0) {
        if (argc < 3) { kprintf("Usage: pkg remove <name>\n"); return 1; }
        int r = modelpkg_uninstall(argv[2]);
        kprintf("%s '%s'\n", r == 0 ? "Removed" : "Failed to remove", argv[2]);
        return r;
    }

    if (shell_strcmp(argv[1], "info") == 0) {
        if (argc < 3) { kprintf("Usage: pkg info <name>\n"); return 1; }
        model_manifest_t m;
        int r = modelpkg_info(argv[2], &m);
        if (r == 0) {
            kprintf("\n  Name:    %s\n  Version: %s\n  Size:    %lu bytes\n",
                    m.name, m.version, m.total_size);
            kprintf("  Arch:    %s\n  Format:  %s\n  Params:  %lu\n\n",
                    m.architecture, m.format, m.param_count);
        } else {
            kprintf("Package '%s' not found.\n", argv[2]);
        }
        return r;
    }

    if (shell_strcmp(argv[1], "update") == 0) {
        if (argc >= 3) {
            kprintf("[PKG] Updating '%s'...\n", argv[2]);
            return modelpkg_update(argv[2]);
        }
        kprintf("[PKG] Updating all...\n");
        return modelpkg_update_all();
    }

    if (shell_strcmp(argv[1], "verify") == 0) {
        if (argc < 3) { kprintf("Usage: pkg verify <name>\n"); return 1; }
        int r = modelpkg_verify(argv[2]);
        kprintf("[PKG] '%s': %s\n", argv[2], r == 0 ? "VERIFIED OK" : "CORRUPT");
        return r;
    }

    kprintf("Unknown pkg subcommand: %s\n", argv[1]);
    return 1;
}

/* =============================================================================
 * SANDBOX — wired to real API
 * =============================================================================*/

static int cmd_sandbox(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: sandbox <create|destroy> [args...]\n");
        kprintf("  create <name> <strict|standard|permissive>\n");
        kprintf("  destroy <id>\n");
        return 1;
    }

    if (shell_strcmp(argv[1], "create") == 0) {
        if (argc < 4) {
            kprintf("Usage: sandbox create <name> <strict|standard|permissive>\n");
            return 1;
        }
        sandbox_policy_t policy = SANDBOX_POLICY_STANDARD;
        if (shell_strcmp(argv[3], "strict") == 0) policy = SANDBOX_POLICY_STRICT;
        else if (shell_strcmp(argv[3], "permissive") == 0) policy = SANDBOX_POLICY_PERMISSIVE;

        sandbox_t *sb = sandbox_create(argv[2], policy);
        if (sb) {
            sandbox_activate(sb->id);
            kprintf("Created sandbox #%lu '%s' (%s)\n", sb->id, argv[2], argv[3]);
        } else {
            kprintf("Failed to create sandbox.\n");
        }
        return sb ? 0 : 1;
    }

    if (shell_strcmp(argv[1], "destroy") == 0) {
        if (argc < 3) { kprintf("Usage: sandbox destroy <id>\n"); return 1; }
        uint64_t id = shell_atou64(argv[2]);
        int r = sandbox_destroy(id);
        kprintf("Sandbox #%lu %s\n", id, r == 0 ? "destroyed" : "not found");
        return r;
    }

    kprintf("Unknown sandbox subcommand: %s\n", argv[1]);
    return 1;
}

/* =============================================================================
 * DEMO & BENCH COMMANDS
 * =============================================================================*/

static int cmd_bench(aishell_t *sh, int argc, char **argv)
{
    extern void run_benchmarks(void);
    kprintf("[BENCH] Running performance benchmarks...\n\n");
    run_benchmarks();
    return 0;
}

static int cmd_demo(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: demo <type>\n");
        kprintf("  infer       Neural network inference demos\n");
        kprintf("  quant       INT16 quantized inference\n");
        kprintf("  evolve      Neuroevolution architecture search\n");
        kprintf("  train       Backprop + Adam training\n");
        kprintf("  sne         Speculative Neural Execution\n");
        kprintf("  transformer KV-cache transformer engine\n");
        kprintf("  q4          INT4 block quantization\n");
        kprintf("  arena       Tensor memory arena\n");
        kprintf("  mathllm     Micro math LLMs\n");
        kprintf("  gguf        GGUF model parser demo\n");
        kprintf("  llmeval     Full LLM benchmark\n");
        kprintf("  smp         SMP multi-core demos\n");
        kprintf("  all         Run everything\n");
        return 1;
    }
    extern void nn_run_demos(void);
    extern void nn_quant_demos(void);
    extern void nn_evolve_demos(void);
    extern void nn_train_demos(void);
    extern void sne_run_demos(void);
    extern void tf_run_demos(void);
    extern void q4_run_demos(void);
    extern void arena_run_demos(void);
    extern void math_llm_run_eval(void);
    extern void gguf_run_demos(void);
    extern void llm_run_full_eval(void);

    if (shell_strcmp(argv[1], "infer") == 0)      { nn_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "quant") == 0)       { nn_quant_demos(); return 0; }
    if (shell_strcmp(argv[1], "evolve") == 0)      { nn_evolve_demos(); return 0; }
    if (shell_strcmp(argv[1], "train") == 0)       { nn_train_demos(); return 0; }
    if (shell_strcmp(argv[1], "sne") == 0)         { sne_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "transformer") == 0) { tf_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "q4") == 0)          { q4_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "arena") == 0)       { arena_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "mathllm") == 0)     { math_llm_run_eval(); return 0; }
    if (shell_strcmp(argv[1], "gguf") == 0)        { gguf_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "llmeval") == 0)     { llm_run_full_eval(); return 0; }
    if (shell_strcmp(argv[1], "smp") == 0)         { smp_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "all") == 0) {
        nn_run_demos(); nn_quant_demos(); nn_evolve_demos();
        nn_train_demos(); sne_run_demos(); tf_run_demos();
        q4_run_demos(); arena_run_demos(); math_llm_run_eval();
        gguf_run_demos(); smp_run_demos();
        return 0;
    }

    kprintf("Unknown demo: %s (type 'demo' for list)\n", argv[1]);
    return 1;
}

/* =============================================================================
 * SELFTEST
 * =============================================================================*/

static int cmd_selftest(aishell_t *sh, int argc, char **argv)
{
    extern void selftest_run_all(void);
    kprintf("[TEST] Running production self-test suite...\n\n");
    selftest_run_all();
    return 0;
}

/* =============================================================================
 * OTA / FLASH / RUN
 * =============================================================================*/

static int cmd_ota(aishell_t *sh, int argc, char **argv)
{
    kprintf("[OTA] Entering chain-load receive mode\n");
    int r = ota_receive_and_chainload();
    kprintf("[OTA] Failed (error %d)\n", r);
    return r;
}

static int cmd_flash(aishell_t *sh, int argc, char **argv)
{
    kprintf("[FLASH] Entering persistent flash mode\n");
    int r = ota_receive_and_flash();
    kprintf("[FLASH] Failed (error %d)\n", r);
    return r;
}

static int cmd_run(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: run <script.pseudo>\n");
        return 1;
    }

    int fd = tfs_open(argv[1], 0);
    if (fd < 0) {
        kprintf("run: cannot open '%s'\n", argv[1]);
        return 1;
    }

    tfs_inode_t info;
    if (tfs_stat(argv[1], &info) < 0 || info.size == 0) {
        tfs_close(fd);
        kprintf("run: '%s' is empty or not found\n", argv[1]);
        return 1;
    }

    uint64_t sz = info.size < 4096 ? info.size : 4096;
    static char script_buf[4096];
    int n = tfs_read(fd, script_buf, sz, 0);
    tfs_close(fd);

    if (n <= 0) {
        kprintf("run: failed to read '%s'\n", argv[1]);
        return 1;
    }
    script_buf[n] = '\0';

    kprintf("[JIT] Executing %s (%d bytes)...\n", argv[1], n);
    if (sh->runtime)
        pseudo_exec_string(sh->runtime, script_buf);
    return 0;
}

/* =============================================================================
 * EXTENDED FILESYSTEM COMMANDS
 * =============================================================================*/

static int cmd_pwd(aishell_t *sh, int argc, char **argv)
{
    kprintf("%s\n", sh->cwd[0] ? sh->cwd : "/");
    return 0;
}

static int cmd_cd(aishell_t *sh, int argc, char **argv)
{
    const char *path = argc >= 2 ? argv[1] : "/";
    if (shell_strcmp(path, "/") != 0) {
        tfs_inode_t info;
        if (tfs_stat(path, &info) < 0) {
            kprintf("cd: %s: no such directory\n", path);
            return 1;
        }
    }
    shell_strncpy(sh->cwd, path, SHELL_MAX_PATH);
    return 0;
}

static int cmd_cp(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) { kprintf("Usage: cp <src> <dst>\n"); return 1; }
    tfs_inode_t info;
    if (tfs_stat(argv[1], &info) < 0) { kprintf("cp: %s: not found\n", argv[1]); return 1; }
    int sfd = tfs_open(argv[1], 0);
    if (sfd < 0) { kprintf("cp: cannot open %s\n", argv[1]); return 1; }
    tfs_create(argv[2], info.type);
    int dfd = tfs_open(argv[2], 1);
    if (dfd < 0) { tfs_close(sfd); kprintf("cp: cannot create %s\n", argv[2]); return 1; }
    static char cpbuf[4096];
    uint64_t off = 0;
    while (off < info.size) {
        uint64_t chunk = info.size - off;
        if (chunk > sizeof(cpbuf)) chunk = sizeof(cpbuf);
        int n = tfs_read(sfd, cpbuf, chunk, off);
        if (n <= 0) break;
        tfs_write(dfd, cpbuf, n, off);
        off += n;
    }
    tfs_close(sfd); tfs_close(dfd);
    kprintf("Copied %s -> %s (%lu bytes)\n", argv[1], argv[2], off);
    return 0;
}

static int cmd_mv(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) { kprintf("Usage: mv <src> <dst>\n"); return 1; }
    /* Implement as copy + remove (no native rename in TFS) */
    int r = cmd_cp(sh, argc, argv);
    if (r != 0) return r;
    tfs_unlink(argv[1]);
    kprintf("Moved %s -> %s\n", argv[1], argv[2]);
    return 0;
}

static int cmd_head(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) { kprintf("Usage: head <file> [lines]\n"); return 1; }
    int nlines = argc >= 3 ? shell_atoi(argv[2]) : 10;
    if (nlines <= 0) nlines = 10;
    tfs_inode_t info;
    if (tfs_stat(argv[1], &info) < 0) { kprintf("head: %s: not found\n", argv[1]); return 1; }
    int fd = tfs_open(argv[1], 0);
    if (fd < 0) { kprintf("head: cannot open %s\n", argv[1]); return 1; }
    static char hbuf[4096];
    uint64_t tr = info.size < sizeof(hbuf) - 1 ? info.size : sizeof(hbuf) - 1;
    int n = tfs_read(fd, hbuf, tr, 0); tfs_close(fd);
    if (n <= 0) return 0;
    hbuf[n] = '\0';
    int shown = 0;
    for (int i = 0; i < n && shown < nlines; i++) {
        char ch[2] = {hbuf[i], 0}; kprintf("%s", ch);
        if (hbuf[i] == '\n') shown++;
    }
    if (shown == 0 || hbuf[n - 1] != '\n') kprintf("\n");
    return 0;
}

static int cmd_tail(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) { kprintf("Usage: tail <file> [lines]\n"); return 1; }
    int nlines = argc >= 3 ? shell_atoi(argv[2]) : 10;
    if (nlines <= 0) nlines = 10;
    tfs_inode_t info;
    if (tfs_stat(argv[1], &info) < 0) { kprintf("tail: %s: not found\n", argv[1]); return 1; }
    int fd = tfs_open(argv[1], 0);
    if (fd < 0) { kprintf("tail: cannot open %s\n", argv[1]); return 1; }
    static char tbuf[4096];
    uint64_t tr = info.size < sizeof(tbuf) - 1 ? info.size : sizeof(tbuf) - 1;
    uint64_t offset = info.size > tr ? info.size - tr : 0;
    int n = tfs_read(fd, tbuf, tr, offset); tfs_close(fd);
    if (n <= 0) return 0;
    tbuf[n] = '\0';
    int count = 0, start = n;
    for (int i = n - 1; i >= 0; i--) {
        if (tbuf[i] == '\n') { count++; if (count > nlines) { start = i + 1; break; } }
    }
    if (count <= nlines) start = 0;
    kprintf("%s", &tbuf[start]);
    if (n > 0 && tbuf[n - 1] != '\n') kprintf("\n");
    return 0;
}

static int cmd_wc(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) { kprintf("Usage: wc <file>\n"); return 1; }
    tfs_inode_t info;
    if (tfs_stat(argv[1], &info) < 0) { kprintf("wc: %s: not found\n", argv[1]); return 1; }
    int fd = tfs_open(argv[1], 0);
    if (fd < 0) { kprintf("wc: cannot open %s\n", argv[1]); return 1; }
    static char wbuf[4096];
    uint64_t tr = info.size < sizeof(wbuf) - 1 ? info.size : sizeof(wbuf) - 1;
    int n = tfs_read(fd, wbuf, tr, 0); tfs_close(fd);
    int lines = 0, words = 0, in_word = 0;
    for (int i = 0; i < n; i++) {
        if (wbuf[i] == '\n') lines++;
        if (wbuf[i] == ' ' || wbuf[i] == '\n' || wbuf[i] == '\t') in_word = 0;
        else if (!in_word) { in_word = 1; words++; }
    }
    kprintf("  %d lines  %d words  %lu bytes  %s\n", lines, words, info.size, argv[1]);
    return 0;
}

static int cmd_grep(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) { kprintf("Usage: grep <pattern> <file>\n"); return 1; }
    tfs_inode_t info;
    if (tfs_stat(argv[2], &info) < 0) { kprintf("grep: %s: not found\n", argv[2]); return 1; }
    int fd = tfs_open(argv[2], 0);
    if (fd < 0) { kprintf("grep: cannot open %s\n", argv[2]); return 1; }
    static char gbuf[4096];
    uint64_t tr = info.size < sizeof(gbuf) - 1 ? info.size : sizeof(gbuf) - 1;
    int n = tfs_read(fd, gbuf, tr, 0); tfs_close(fd);
    gbuf[n] = '\0';
    const char *pat = argv[1];
    int plen = (int)shell_strlen(pat);
    int line_num = 1, matches = 0;
    char *ls = gbuf;
    for (int i = 0; i <= n; i++) {
        if (gbuf[i] == '\n' || gbuf[i] == '\0') {
            char sv = gbuf[i]; gbuf[i] = '\0';
            char *p = ls; int found = 0;
            while (*p) {
                int m = 1;
                for (int j = 0; j < plen; j++) { if (p[j] == 0 || p[j] != pat[j]) { m = 0; break; } }
                if (m) { found = 1; break; }
                p++;
            }
            if (found) { kprintf("%d: %s\n", line_num, ls); matches++; }
            gbuf[i] = sv; ls = &gbuf[i + 1]; line_num++;
        }
    }
    if (matches == 0) kprintf("(no matches)\n");
    return matches > 0 ? 0 : 1;
}

static int cmd_find(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) { kprintf("Usage: find <pattern> [path]\n"); return 1; }
    const char *path = argc >= 3 ? argv[2] : "/";
    tfs_inode_t entries[64];
    uint32_t count = 0;
    if (tfs_readdir(path, entries, 64, &count) < 0) { kprintf("find: cannot read %s\n", path); return 1; }
    const char *pat = argv[1];
    int plen = (int)shell_strlen(pat), found = 0;
    for (uint32_t i = 0; i < count; i++) {
        const char *name = entries[i].name;
        int nlen = (int)shell_strlen(name);
        for (int j = 0; j <= nlen - plen; j++) {
            int m = 1;
            for (int k = 0; k < plen; k++) { if (name[j + k] != pat[k]) { m = 0; break; } }
            if (m) { kprintf("  %s/%s\n", path, name); found++; break; }
        }
    }
    kprintf("\n  %d match(es)\n", found);
    return found > 0 ? 0 : 1;
}

static int cmd_df(aishell_t *sh, int argc, char **argv)
{
    mm_stats_t stats;
    tensor_mm_get_stats(&stats);
    uint64_t blk_cap = virtio_blk_capacity();
    kprintf("\n  Filesystem        Size       Used       Free       Use%%\n");
    kprintf("  ---------------   ---------  ---------  ---------  ----\n");
    kprintf("  tensorfs          %6lu MB  %6lu MB  %6lu MB  %3lu%%\n",
            stats.tensor_heap_size / (1024*1024),
            stats.tensor_heap_used / (1024*1024),
            (stats.tensor_heap_size - stats.tensor_heap_used) / (1024*1024),
            stats.tensor_heap_size > 0 ? stats.tensor_heap_used * 100 / stats.tensor_heap_size : 0);
    if (blk_cap > 0)
        kprintf("  virtio-blk        %6lu MB  %6s     %6s     ---\n", blk_cap / (1024*1024), "---", "---");
    kprintf("\n");
    return 0;
}

static int cmd_du(aishell_t *sh, int argc, char **argv)
{
    const char *path = argc >= 2 ? argv[1] : "/";
    tfs_inode_t entries[64];
    uint32_t count = 0;
    if (tfs_readdir(path, entries, 64, &count) < 0) { kprintf("du: %s: not found\n", path); return 1; }
    uint64_t total = 0;
    for (uint32_t i = 0; i < count; i++) {
        total += entries[i].size;
        if (entries[i].size >= 1024*1024) kprintf("  %6lu MB  %s\n", entries[i].size / (1024*1024), entries[i].name);
        else if (entries[i].size >= 1024) kprintf("  %6lu KB  %s\n", entries[i].size / 1024, entries[i].name);
        else kprintf("  %6lu B   %s\n", entries[i].size, entries[i].name);
    }
    if (total >= 1024*1024) kprintf("  --------\n  %lu MB total\n", total / (1024*1024));
    else kprintf("  --------\n  %lu B total\n", total);
    kprintf("\n");
    return 0;
}

static int cmd_tree(aishell_t *sh, int argc, char **argv)
{
    const char *path = argc >= 2 ? argv[1] : "/";
    tfs_inode_t entries[64];
    uint32_t count = 0;
    kprintf("%s\n", path);
    if (tfs_readdir(path, entries, 64, &count) < 0) { kprintf("  (empty)\n"); return 1; }
    for (uint32_t i = 0; i < count; i++) {
        kprintf("%s %s", (i == count - 1) ? "`--" : "|--", entries[i].name);
        if (entries[i].type == TFS_FILE_DIR) kprintf("/");
        kprintf("\n");
    }
    kprintf("\n%u entries\n", count);
    return 0;
}

/* =============================================================================
 * SYSTEM UTILITY COMMANDS
 * =============================================================================*/

static int cmd_whoami(aishell_t *sh, int argc, char **argv) { (void)sh; (void)argc; (void)argv; kprintf("root\n"); return 0; }
static int cmd_hostname(aishell_t *sh, int argc, char **argv) { (void)sh; (void)argc; (void)argv; kprintf("tensoros\n"); return 0; }

static int cmd_date(aishell_t *sh, int argc, char **argv)
{
    extern uint64_t watchdog_uptime_ms(void);
    uint64_t ms = watchdog_uptime_ms();
    uint64_t sec = ms / 1000, min = sec / 60, hr = min / 60;
    kprintf("Up %lu:%02lu:%02lu.%03lu (no RTC)\n", hr, min % 60, sec % 60, ms % 1000);
    return 0;
}

static int cmd_sleep(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) { kprintf("Usage: sleep <ms>\n"); return 1; }
    int ms = shell_atoi(argv[1]);
    if (ms <= 0 || ms > 60000) { kprintf("sleep: 1-60000 ms\n"); return 1; }
    extern uint64_t watchdog_uptime_ms(void);
    uint64_t end = watchdog_uptime_ms() + (uint64_t)ms;
    while (watchdog_uptime_ms() < end) {
#ifdef __aarch64__
        __asm__ volatile ("wfi");
#else
        __asm__ volatile ("hlt");
#endif
    }
    return 0;
}

static int cmd_true_cmd(aishell_t *sh, int argc, char **argv) { (void)sh; (void)argc; (void)argv; return 0; }
static int cmd_false_cmd(aishell_t *sh, int argc, char **argv) { (void)sh; (void)argc; (void)argv; return 1; }

static int cmd_yes(aishell_t *sh, int argc, char **argv)
{
    const char *s = argc >= 2 ? argv[1] : "y";
    for (int i = 0; i < 100; i++) kprintf("%s\n", s);
    return 0;
}

static int cmd_seq(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) { kprintf("Usage: seq <from> <to>\n"); return 1; }
    int from = shell_atoi(argv[1]), to = shell_atoi(argv[2]);
    int step = from <= to ? 1 : -1, cnt = 0;
    for (int i = from; step > 0 ? i <= to : i >= to; i += step)
        if (++cnt > 10000) break; else kprintf("%d\n", i);
    return 0;
}

static uint32_t xorshift_state = 0x12345678;

static int cmd_rand(aishell_t *sh, int argc, char **argv)
{
    xorshift_state ^= (uint32_t)rdtsc_fenced();
    xorshift_state ^= xorshift_state << 13;
    xorshift_state ^= xorshift_state >> 17;
    xorshift_state ^= xorshift_state << 5;
    int max_val = argc >= 2 ? shell_atoi(argv[1]) : 100;
    if (max_val <= 0) max_val = 100;
    kprintf("%u\n", xorshift_state % (uint32_t)max_val);
    return 0;
}

static int cmd_logo(aishell_t *sh, int argc, char **argv)
{
    shell_print_banner();
    return 0;
}

static int cmd_version(aishell_t *sh, int argc, char **argv)
{
    kprintf("TensorOS v0.1.0 \"Neuron\"\n");
    kprintf("Build: bare-metal %s\n",
#ifdef __aarch64__
            "aarch64 NEON"
#else
            cpu_features.avx2_usable ? "x86_64 AVX2+FMA" : "x86_64 SSE2"
#endif
    );
    kprintf("Shell: aishell (readline, %d history slots, tab-complete)\n", SHELL_MAX_HISTORY);
    return 0;
}

static int cmd_about(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n");
    kprintf("  TensorOS - The AI-First Operating System\n");
    kprintf("  ==========================================\n\n");
    kprintf("  A bare-metal OS built from scratch for AI workloads.\n");
    kprintf("  No Linux, no BIOS calls — raw hardware and neural nets.\n\n");
    kprintf("  Features:\n");
    kprintf("    - GGUF model loading & Q8_0 inference engine\n");
    kprintf("    - SSE2/AVX2 SIMD matrix math with JIT compilation\n");
    kprintf("    - TCP/IP stack & OpenAI-compatible API server\n");
    kprintf("    - TensorFS filesystem with model-aware types\n");
    kprintf("    - MEU scheduler for model execution units\n");
    kprintf("    - SMP multi-core scheduling\n");
    kprintf("    - Git version control subsystem\n");
    kprintf("    - Full readline shell with history & tab-complete\n\n");
    kprintf("  https://github.com/NagusameCS/TensorOS\n\n");
    return 0;
}

/* =============================================================================
 * HARDWARE / DEBUG COMMANDS
 * =============================================================================*/

static int cmd_free_mem(aishell_t *sh, int argc, char **argv)
{
    mm_stats_t stats;
    tensor_mm_get_stats(&stats);
    kprintf("             total       used       free\n");
    kprintf("Mem:    %9lu  %9lu  %9lu  KB\n",
            stats.total_phys / 1024, (stats.total_phys - stats.free_phys) / 1024, stats.free_phys / 1024);
    kprintf("Heap:   %9lu  %9lu  %9lu  KB\n",
            stats.tensor_heap_size / 1024, stats.tensor_heap_used / 1024,
            (stats.tensor_heap_size - stats.tensor_heap_used) / 1024);
    kprintf("Cache:  %9lu  %9lu  %9lu  KB\n",
            stats.model_cache_size / 1024, stats.model_cache_used / 1024,
            (stats.model_cache_size - stats.model_cache_used) / 1024);
    return 0;
}

static int cmd_top(aishell_t *sh, int argc, char **argv)
{
    extern uint64_t watchdog_uptime_ms(void);
    mm_stats_t stats;
    tensor_mm_get_stats(&stats);
    uint64_t sched_ops = 0, sched_disp = 0, sched_lat = 0;
    tensor_sched_get_stats(&sched_ops, &sched_disp, &sched_lat);
    kprintf("\ntop - up %lu s, %d cpu(s), %lu MB free\n",
            watchdog_uptime_ms() / 1000, smp.cpu_count, stats.free_phys / (1024*1024));
    kprintf("MEUs: %d,  dispatches: %lu,  tensor ops: %lu\n\n",
            kstate.meu_count, sched_disp, sched_ops);
    kprintf("  PID  MEM MB  STATE      NAME\n");
    kprintf("  ---  ------  ---------  --------------------\n");
    kprintf("    0      --  RUNNING    kernel\n");
    kprintf("    1      --  RUNNING    aishell\n");
    for (uint32_t i = 0; i < kstate.meu_count; i++) {
        model_exec_unit_t *meu = &kstate.meus[i];
        kprintf("  %3u  %6lu  %-9s  %s\n", i + 2, meu->mem_used / (1024*1024),
                meu->state == MEU_STATE_RUNNING ? "RUNNING" :
                meu->state == MEU_STATE_READY   ? "READY"   : "OTHER", meu->name);
    }
    if (llm_is_loaded())
        kprintf("  %3u      --  RUNNING    %s (LLM)\n", kstate.meu_count + 2, llm_model_name());
    kprintf("\n");
    return 0;
}

static int cmd_dmesg(aishell_t *sh, int argc, char **argv)
{
    kprintf("[dmesg] Kernel log buffer not available in this build.\n");
    kprintf("  Use serial output (build/serial.log) for boot messages.\n");
    return 0;
}

static int cmd_irq(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n=== Interrupt Statistics ===\n");
    kprintf("  IRQ0  (PIT/Timer):  active\n");
    kprintf("  IRQ1  (Keyboard):   active\n");
    kprintf("  IRQ14 (Disk):       %s\n", virtio_blk_capacity() > 0 ? "active" : "inactive");
    kprintf("  Vectors 32-255:     available\n\n");
    return 0;
}

static int cmd_peek(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) { kprintf("Usage: peek <addr> [1|2|4|8]\n"); return 1; }
    uint64_t addr = shell_atou64(argv[1]);
    int sz = argc >= 3 ? shell_atoi(argv[2]) : 4;
    switch (sz) {
        case 1:  kprintf("0x%lx = 0x%02x (%u)\n", addr, *(volatile uint8_t *)addr, *(volatile uint8_t *)addr); break;
        case 2:  kprintf("0x%lx = 0x%04x (%u)\n", addr, *(volatile uint16_t *)addr, *(volatile uint16_t *)addr); break;
        case 8:  kprintf("0x%lx = 0x%016lx\n", addr, *(volatile uint64_t *)addr); break;
        default: kprintf("0x%lx = 0x%08x (%u)\n", addr, *(volatile uint32_t *)addr, *(volatile uint32_t *)addr); break;
    }
    return 0;
}

static int cmd_poke(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) { kprintf("Usage: poke <addr> <value> [1|2|4|8]\n"); return 1; }
    uint64_t addr = shell_atou64(argv[1]);
    uint64_t val = shell_atou64(argv[2]);
    int sz = argc >= 4 ? shell_atoi(argv[3]) : 4;
    switch (sz) {
        case 1:  *(volatile uint8_t *)addr = (uint8_t)val; break;
        case 2:  *(volatile uint16_t *)addr = (uint16_t)val; break;
        case 8:  *(volatile uint64_t *)addr = val; break;
        default: *(volatile uint32_t *)addr = (uint32_t)val; break;
    }
    kprintf("Wrote 0x%lx to 0x%lx (%d-byte)\n", val, addr, sz);
    return 0;
}

static int cmd_strings(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) { kprintf("Usage: strings <addr> [len]\n"); return 1; }
    uint64_t addr = shell_atou64(argv[1]);
    int len = argc >= 3 ? shell_atoi(argv[2]) : 256;
    if (len <= 0) len = 256; if (len > 65536) len = 65536;
    const uint8_t *p = (const uint8_t *)addr;
    int run = 0, found = 0;
    for (int i = 0; i < len; i++) {
        if (p[i] >= 0x20 && p[i] < 0x7F) { run++; }
        else {
            if (run >= 4) { for (int j = i - run; j < i; j++) { char ch[2] = {(char)p[j], 0}; kprintf("%s", ch); } kprintf("\n"); found++; }
            run = 0;
        }
    }
    if (run >= 4) { for (int j = len - run; j < len; j++) { char ch[2] = {(char)p[j], 0}; kprintf("%s", ch); } kprintf("\n"); found++; }
    kprintf("(%d strings found)\n", found);
    return 0;
}

static int cmd_crc32(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) { kprintf("Usage: crc32 <addr> <len>\n"); return 1; }
    uint64_t addr = shell_atou64(argv[1]);
    int len = shell_atoi(argv[2]);
    if (len <= 0 || len > 0x1000000) { kprintf("Invalid length\n"); return 1; }
    const uint8_t *p = (const uint8_t *)addr;
    uint32_t crc = 0xFFFFFFFF;
    for (int i = 0; i < len; i++) {
        crc ^= p[i];
        for (int bit = 0; bit < 8; bit++) crc = (crc & 1) ? (crc >> 1) ^ 0xEDB88320 : crc >> 1;
    }
    crc ^= 0xFFFFFFFF;
    kprintf("CRC32: 0x%08x (%u bytes at 0x%lx)\n", crc, len, addr);
    return 0;
}

#ifndef __aarch64__
static int cmd_rdmsr(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) { kprintf("Usage: rdmsr <msr>\n"); return 1; }
    uint32_t msr = (uint32_t)shell_atou64(argv[1]);
    uint32_t lo, hi;
    __asm__ volatile ("rdmsr" : "=a"(lo), "=d"(hi) : "c"(msr));
    kprintf("MSR 0x%x = 0x%08x%08x\n", msr, hi, lo);
    return 0;
}

static int cmd_wrmsr(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) { kprintf("Usage: wrmsr <msr> <value>\n"); return 1; }
    uint32_t msr = (uint32_t)shell_atou64(argv[1]);
    uint64_t val = shell_atou64(argv[2]);
    uint32_t lo = (uint32_t)val, hi = (uint32_t)(val >> 32);
    __asm__ volatile ("wrmsr" : : "c"(msr), "a"(lo), "d"(hi));
    kprintf("Wrote 0x%08x%08x to MSR 0x%x\n", hi, lo, msr);
    return 0;
}

static int cmd_cpuid_cmd(aishell_t *sh, int argc, char **argv)
{
    uint32_t leaf = argc >= 2 ? (uint32_t)shell_atou64(argv[1]) : 0;
    uint32_t eax, ebx, ecx, edx;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(leaf), "c"(0));
    kprintf("CPUID leaf 0x%x: eax=0x%08x ebx=0x%08x ecx=0x%08x edx=0x%08x\n", leaf, eax, ebx, ecx, edx);
    return 0;
}

static int cmd_ioread(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) { kprintf("Usage: ioread <port>\n"); return 1; }
    uint16_t port = (uint16_t)shell_atou64(argv[1]);
    kprintf("I/O port 0x%x = 0x%02x\n", port, inb(port));
    return 0;
}

static int cmd_iowrite(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) { kprintf("Usage: iowrite <port> <value>\n"); return 1; }
    uint16_t port = (uint16_t)shell_atou64(argv[1]);
    uint8_t val = (uint8_t)shell_atou64(argv[2]);
    outb(port, val);
    kprintf("Wrote 0x%02x to I/O port 0x%x\n", val, port);
    return 0;
}
#endif /* !__aarch64__ */

/* =============================================================================
 * SHELL UTILITY COMMANDS
 * =============================================================================*/

static int cmd_repeat(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) { kprintf("Usage: repeat <n> <command> [args...]\n"); return 1; }
    int n = shell_atoi(argv[1]);
    if (n <= 0 || n > 10000) { kprintf("repeat: 1-10000\n"); return 1; }
    static char subcmd[SHELL_MAX_LINE];
    int pos = 0;
    for (int i = 2; i < argc && pos < SHELL_MAX_LINE - 2; i++) {
        if (i > 2) subcmd[pos++] = ' ';
        for (const char *p = argv[i]; *p && pos < SHELL_MAX_LINE - 2; p++) subcmd[pos++] = *p;
    }
    subcmd[pos] = '\0';
    for (int i = 0; i < n; i++) {
        int sa; char *sv[SHELL_MAX_ARGS];
        shell_parse_line(subcmd, &sa, sv);
        shell_exec_builtin(sh, sa, sv);
    }
    return 0;
}

static int cmd_alias(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        if (sh->alias_count == 0) { kprintf("No aliases defined.\n"); return 0; }
        for (int i = 0; i < sh->alias_count; i++)
            kprintf("  %s = '%s'\n", sh->aliases[i].name, sh->aliases[i].command);
        return 0;
    }
    const char *name = argv[1];
    static char name_buf[32];
    const char *cmd_str = "";
    int eq = -1;
    for (int i = 0; name[i] && i < 31; i++) if (name[i] == '=') { eq = i; break; }
    if (eq > 0) {
        for (int i = 0; i < eq && i < 31; i++) name_buf[i] = name[i];
        name_buf[eq] = '\0'; name = name_buf;
        cmd_str = &argv[1][eq + 1];
    } else if (argc >= 3) {
        static char abuf[SHELL_MAX_LINE]; int pos = 0;
        for (int i = 2; i < argc && pos < SHELL_MAX_LINE - 2; i++) {
            if (i > 2) abuf[pos++] = ' ';
            for (const char *p = argv[i]; *p && pos < SHELL_MAX_LINE - 2; p++) abuf[pos++] = *p;
        }
        abuf[pos] = '\0'; cmd_str = abuf;
    }
    if (shell_strlen(cmd_str) == 0) {
        for (int i = 0; i < sh->alias_count; i++)
            if (shell_strcmp(sh->aliases[i].name, name) == 0) { kprintf("  %s = '%s'\n", sh->aliases[i].name, sh->aliases[i].command); return 0; }
        kprintf("alias: %s: not found\n", name); return 1;
    }
    for (int i = 0; i < sh->alias_count; i++) {
        if (shell_strcmp(sh->aliases[i].name, name) == 0) { shell_strncpy(sh->aliases[i].command, cmd_str, SHELL_MAX_LINE); return 0; }
    }
    if (sh->alias_count >= SHELL_MAX_ALIASES) { kprintf("alias: max reached\n"); return 1; }
    shell_strncpy(sh->aliases[sh->alias_count].name, name, 32);
    shell_strncpy(sh->aliases[sh->alias_count].command, cmd_str, SHELL_MAX_LINE);
    sh->alias_count++;
    kprintf("alias %s='%s'\n", name, cmd_str);
    return 0;
}

static int cmd_unalias(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) { kprintf("Usage: unalias <name>\n"); return 1; }
    for (int i = 0; i < sh->alias_count; i++) {
        if (shell_strcmp(sh->aliases[i].name, argv[1]) == 0) {
            for (int j = i; j < sh->alias_count - 1; j++) sh->aliases[j] = sh->aliases[j + 1];
            sh->alias_count--;
            kprintf("Removed alias: %s\n", argv[1]);
            return 0;
        }
    }
    kprintf("unalias: %s: not found\n", argv[1]);
    return 1;
}

static int cmd_set(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        if (sh->env_count == 0) { kprintf("No variables set.\n"); return 0; }
        for (int i = 0; i < sh->env_count; i++) kprintf("  %s=%s\n", sh->env[i].name, sh->env[i].value);
        return 0;
    }
    const char *name = argv[1]; const char *val = "";
    static char nbuf[32]; int eq = -1;
    for (int i = 0; name[i] && i < 31; i++) if (name[i] == '=') { eq = i; break; }
    if (eq > 0) { for (int i = 0; i < eq && i < 31; i++) nbuf[i] = name[i]; nbuf[eq] = '\0'; name = nbuf; val = &argv[1][eq + 1]; }
    else if (argc >= 3) val = argv[2];
    for (int i = 0; i < sh->env_count; i++)
        if (shell_strcmp(sh->env[i].name, name) == 0) { shell_strncpy(sh->env[i].value, val, 128); return 0; }
    if (sh->env_count >= SHELL_MAX_ENV) { kprintf("set: max vars reached\n"); return 1; }
    shell_strncpy(sh->env[sh->env_count].name, name, 32);
    shell_strncpy(sh->env[sh->env_count].value, val, 128);
    sh->env_count++;
    return 0;
}

static int cmd_env(aishell_t *sh, int argc, char **argv)
{
    return cmd_set(sh, 1, argv);
}

/* =============================================================================
 * NETWORK EXTENDED COMMANDS
 * =============================================================================*/

static int cmd_ping(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) { kprintf("Usage: ping <ip>\n"); return 1; }
    kprintf("PING %s ... (bare-metal ICMP not implemented, use 'net' for stats)\n", argv[1]);
    return 0;
}

static int cmd_ifconfig(aishell_t *sh, int argc, char **argv)
{
    const net_config_t *cfg = netstack_get_config();
    kprintf("\nvirtio0:\n");
    kprintf("  inet %u.%u.%u.%u  netmask 255.255.255.0\n", cfg->ip[0], cfg->ip[1], cfg->ip[2], cfg->ip[3]);
    kprintf("  ether %02x:%02x:%02x:%02x:%02x:%02x\n", cfg->mac[0], cfg->mac[1], cfg->mac[2], cfg->mac[3], cfg->mac[4], cfg->mac[5]);
    kprintf("  gateway %u.%u.%u.%u\n", cfg->gateway[0], cfg->gateway[1], cfg->gateway[2], cfg->gateway[3]);
    kprintf("  HTTP port %u  %s\n\n", cfg->http_port, cfg->server_running ? "LISTENING" : "down");
    return 0;
}

static int cmd_netstat(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n=== Network Connections ===\n");
    netstack_print_stats();
    kprintf("  HTTP API: %s\n\n", netstack_server_running() ? "LISTENING :8080" : "not running");
    return 0;
}

/* =============================================================================
 * DANGEROUS / DEBUG COMMANDS
 * =============================================================================*/

static int cmd_panic(aishell_t *sh, int argc, char **argv)
{
    kprintf("[PANIC] Triggering test kernel panic...\n");
    volatile int z = 0;
    volatile int x = 1 / z;
    (void)x;
    return 0;
}

static int cmd_halt(aishell_t *sh, int argc, char **argv)
{
    kprintf("System halted.\n");
#ifdef __aarch64__
    __asm__ volatile ("msr daifset, #0xF"); /* mask all interrupts */
    while(1) __asm__ volatile ("wfi");
#else
    __asm__ volatile ("cli; hlt");
#endif
    return 0;
}

/* =============================================================================
 * SSH & SECURITY COMMANDS
 * =============================================================================*/

static int cmd_sshd(aishell_t *sh, int argc, char **argv)
{
    (void)sh;
    if (argc < 2) {
        kprintf("Usage: sshd <start|stop|status|keygen|sessions>\n");
        return 0;
    }
    if (shell_strcmp(argv[1], "start") == 0) {
        ssh_server_init();
        ssh_server_start();
    } else if (shell_strcmp(argv[1], "stop") == 0) {
        ssh_server_stop();
    } else if (shell_strcmp(argv[1], "status") == 0) {
        ssh_print_status();
    } else if (shell_strcmp(argv[1], "keygen") == 0) {
        ssh_regenerate_host_key();
    } else if (shell_strcmp(argv[1], "sessions") == 0) {
        kprintf("Active SSH sessions: %d\n", ssh_active_sessions());
    } else if (shell_strcmp(argv[1], "disconnect") == 0) {
        if (argc < 3) { kprintf("Usage: sshd disconnect <id>\n"); return 1; }
        int id = 0;
        const char *p = argv[2];
        while (*p >= '0' && *p <= '9') id = id * 10 + (*p++ - '0');
        ssh_disconnect_session(id, "Disconnected by admin");
    } else {
        kprintf("Unknown sshd subcommand: %s\n", argv[1]);
    }
    return 0;
}

static int cmd_users(aishell_t *sh, int argc, char **argv)
{
    (void)sh;
    if (argc < 2) {
        /* List users */
        ssh_user_t users[SSH_MAX_USERS];
        uint32_t count;
        ssh_list_users(users, SSH_MAX_USERS, &count);
        kprintf("SSH Users (%u):\n", count);
        for (uint32_t i = 0; i < count; i++) {
            kprintf("  %-16s perms=0x%x pubkey=%s\n",
                    users[i].username, users[i].permissions,
                    users[i].pubkey_set ? "yes" : "no");
        }
        return 0;
    }
    if (shell_strcmp(argv[1], "add") == 0) {
        if (argc < 4) { kprintf("Usage: users add <name> <password>\n"); return 1; }
        if (ssh_add_user(argv[2], argv[3], 0x1) == 0)
            kprintf("User '%s' created\n", argv[2]);
        else
            kprintf("Failed to create user\n");
    } else if (shell_strcmp(argv[1], "del") == 0) {
        if (argc < 3) { kprintf("Usage: users del <name>\n"); return 1; }
        if (ssh_remove_user(argv[2]) == 0)
            kprintf("User '%s' removed\n", argv[2]);
        else
            kprintf("User not found\n");
    } else {
        kprintf("Usage: users [add <name> <pass> | del <name>]\n");
    }
    return 0;
}

static int cmd_passwd(aishell_t *sh, int argc, char **argv)
{
    (void)sh;
    if (argc < 3) {
        kprintf("Usage: passwd <username> <new_password>\n");
        return 1;
    }
    if (ssh_change_password(argv[1], argv[2]) == 0)
        kprintf("Password changed for '%s'\n", argv[1]);
    else
        kprintf("User not found\n");
    return 0;
}

static int cmd_firewall(aishell_t *sh, int argc, char **argv)
{
    (void)sh;
    if (argc < 2) {
        fw_print_rules();
        return 0;
    }
    if (shell_strcmp(argv[1], "enable") == 0) {
        fw_enable();
    } else if (shell_strcmp(argv[1], "disable") == 0) {
        fw_disable();
    } else if (shell_strcmp(argv[1], "rules") == 0) {
        fw_print_rules();
    } else if (shell_strcmp(argv[1], "stats") == 0) {
        fw_print_stats();
    } else if (shell_strcmp(argv[1], "flush") == 0) {
        fw_flush_rules();
    } else if (shell_strcmp(argv[1], "allow") == 0 ||
               shell_strcmp(argv[1], "deny") == 0) {
        /* firewall allow|deny <proto> <port> [desc] */
        if (argc < 4) {
            kprintf("Usage: firewall %s <tcp|udp|any> <port> [desc]\n", argv[1]);
            return 1;
        }
        fw_rule_t rule;
        kmemset(&rule, 0, sizeof(rule));
        rule.direction = FW_DIR_INBOUND;
        rule.priority = 100;

        if (shell_strcmp(argv[2], "tcp") == 0) rule.protocol = FW_PROTO_TCP;
        else if (shell_strcmp(argv[2], "udp") == 0) rule.protocol = FW_PROTO_UDP;
        else rule.protocol = FW_PROTO_ANY;

        int port = 0;
        const char *p = argv[3];
        while (*p >= '0' && *p <= '9') port = port * 10 + (*p++ - '0');
        rule.dst_port_min = (uint16_t)port;
        rule.dst_port_max = (uint16_t)port;

        rule.action = (shell_strcmp(argv[1], "allow") == 0) ?
                       FW_ACTION_ALLOW : FW_ACTION_DENY;

        if (argc >= 5) {
            int i;
            for (i = 0; i < 63 && argv[4][i]; i++)
                rule.description[i] = argv[4][i];
            rule.description[i] = '\0';
        }

        if (fw_add_rule(&rule) == 0)
            kprintf("Rule added\n");
        else
            kprintf("Failed to add rule (max %d)\n", MAX_FW_RULES);
    } else {
        kprintf("Usage: firewall <enable|disable|rules|stats|flush|allow|deny>\n");
    }
    return 0;
}

static int cmd_audit(aishell_t *sh, int argc, char **argv)
{
    (void)sh;
    uint32_t n = 20;
    if (argc >= 2) {
        n = 0;
        const char *p = argv[1];
        while (*p >= '0' && *p <= '9') n = n * 10 + (*p++ - '0');
        if (n == 0) {
            if (shell_strcmp(argv[1], "verify") == 0) {
                if (sec_audit_verify_chain() == 0)
                    kprintf("Audit chain integrity: VERIFIED\n");
                else
                    kprintf("Audit chain integrity: COMPROMISED!\n");
                return 0;
            }
            n = 20;
        }
    }
    sec_audit_print(n);
    return 0;
}

static int cmd_keystore(aishell_t *sh, int argc, char **argv)
{
    (void)sh;
    if (argc < 2) {
        keystore_list();
        return 0;
    }
    if (shell_strcmp(argv[1], "list") == 0) {
        keystore_list();
    } else if (shell_strcmp(argv[1], "gen") == 0) {
        if (argc < 3) { kprintf("Usage: keystore gen <name> [type]\n"); return 1; }
        uint8_t key[32];
        crypto_random(key, 32);
        key_type_t type = KEY_TYPE_AES256;
        if (argc >= 4) {
            if (shell_strcmp(argv[3], "hmac") == 0) type = KEY_TYPE_HMAC;
            else if (shell_strcmp(argv[3], "ed25519") == 0) type = KEY_TYPE_ED25519;
            else if (shell_strcmp(argv[3], "x25519") == 0) type = KEY_TYPE_X25519;
        }
        if (keystore_store(argv[2], type, key, 32, 0) == 0)
            kprintf("Key '%s' generated and stored\n", argv[2]);
        else
            kprintf("Failed to store key\n");
        crypto_wipe(key, 32);
    } else if (shell_strcmp(argv[1], "del") == 0) {
        if (argc < 3) { kprintf("Usage: keystore del <name>\n"); return 1; }
        if (keystore_delete(argv[2]) == 0)
            kprintf("Key '%s' deleted\n", argv[2]);
        else
            kprintf("Key not found\n");
    } else {
        kprintf("Usage: keystore [list|gen <name> [type]|del <name>]\n");
    }
    return 0;
}

static int cmd_integrity(aishell_t *sh, int argc, char **argv)
{
    (void)sh;
    (void)argc; (void)argv;
    integrity_verify_all();
    return 0;
}

static int cmd_sec_init(aishell_t *sh, int argc, char **argv)
{
    (void)sh; (void)argc; (void)argv;
    security_init();
    return 0;
}

static int cmd_sha256(aishell_t *sh, int argc, char **argv)
{
    (void)sh;
    if (argc < 2) { kprintf("Usage: sha256 <string>\n"); return 1; }
    uint8_t hash[32];
    sha256((const uint8_t *)argv[1], kstrlen(argv[1]), hash);
    kprintf("SHA-256: ");
    for (int i = 0; i < 32; i++)
        kprintf("%02x", hash[i]);
    kprintf("\n");
    return 0;
}

/* =============================================================================
 * HELP
 * =============================================================================*/

static void shell_print_help(void)
{
    kprintf("\n");
    kprintf("TensorOS AI Shell - Commands (%d+ builtins)\n", 80);
    kprintf("============================================\n");

    kprintf("\n  --- General ---\n");
    kprintf("  help                      This message\n");
    kprintf("  status / sysinfo          System overview / full info\n");
    kprintf("  uname [-a]               OS version\n");
    kprintf("  uptime / date             System uptime\n");
    kprintf("  clear                     Clear screen\n");
    kprintf("  reboot / halt             Restart / halt CPU\n");
    kprintf("  exit / quit               Shutdown\n");
    kprintf("  whoami / hostname         User / hostname\n");
    kprintf("  version / about / logo    Version info / about / banner\n");

    kprintf("\n  --- Filesystem ---\n");
    kprintf("  ls [path]                 List directory\n");
    kprintf("  cat <file>                Print file contents\n");
    kprintf("  head/tail <file> [n]      First/last N lines\n");
    kprintf("  stat <path>               Inode information\n");
    kprintf("  mkdir/touch/rm <path>     Create dir/file, remove\n");
    kprintf("  write <file> <text>       Write text to file\n");
    kprintf("  cp <src> <dst>            Copy file\n");
    kprintf("  mv <src> <dst>            Move/rename file\n");
    kprintf("  wc <file>                 Line/word/byte count\n");
    kprintf("  grep <pat> <file>         Search in file\n");
    kprintf("  find <pat> [path]         Find files by name\n");
    kprintf("  pwd / cd <dir>            Working directory\n");
    kprintf("  df / du [path]            Disk free / usage\n");
    kprintf("  tree [path]               Directory tree\n");

    kprintf("\n  --- Hardware & System ---\n");
    kprintf("  cpu                       CPU features & cores\n");
    kprintf("  mem [defrag]              Memory stats\n");
    kprintf("  free                      Compact memory summary\n");
    kprintf("  top                       Process/MEU overview\n");
    kprintf("  disk                      Block device info\n");
    kprintf("  smp                       Multi-core status\n");
    kprintf("  net / ifconfig / netstat  Network info\n");
    kprintf("  ps / lspci                Processes / PCI devices\n");
    kprintf("  monitor                   System monitor\n");
    kprintf("  sched [balance|flush]     Scheduler control\n");
    kprintf("  jit [test]                JIT compiler stats\n");
    kprintf("  irq / dmesg               Interrupt stats / kernel log\n");

    kprintf("\n  --- AI / LLM ---\n");
    kprintf("  llm <prompt>              Chat with loaded LLM\n");
    kprintf("  ai <prompt>               Alias for 'llm'\n");
    kprintf("  reset                     Reset LLM context\n");
    kprintf("  infer <prompt>            Inference (alias for llm)\n");
    kprintf("  train <demo|xor>          Training demos\n");
    kprintf("  deploy <model> [port]     Deploy as HTTP service\n");
    kprintf("  serve                     Start OpenAI-compat API server\n");

    kprintf("\n  --- Model & Tensor ---\n");
    kprintf("  model list|create|load|info|kill   MEU management\n");
    kprintf("  tensor selftest|matmul|alloc|free  Tensor ops\n");

    kprintf("\n  --- Developer Tools ---\n");
    kprintf("  hexdump <addr> [len]      Hex dump memory\n");
    kprintf("  peek <addr> [1|2|4|8]     Read memory\n");
    kprintf("  poke <addr> <val> [sz]    Write memory\n");
    kprintf("  strings <addr> [len]      Extract ASCII strings\n");
    kprintf("  crc32 <addr> <len>        Compute CRC32\n");
    kprintf("  calc <a> <op> <b>         Integer calculator\n");
    kprintf("  time <command>            Time a command\n");
    kprintf("  repeat <n> <cmd>          Run command N times\n");
    kprintf("  echo / sleep <ms>         Print text / delay\n");
    kprintf("  history                   Command history\n");
    kprintf("  selftest / bench          Self-test / benchmarks\n");
    kprintf("  demo <type>               AI demos\n");

    kprintf("\n  --- Shell Features ---\n");
    kprintf("  alias <name>=<cmd>        Create alias\n");
    kprintf("  unalias <name>            Remove alias\n");
    kprintf("  set <name>=<val>          Set variable\n");
    kprintf("  env / export              List/set variables\n");
    kprintf("  rand [max] / seq <a> <b>  Random number / sequence\n");
    kprintf("  true / false / yes [str]  Exit codes / repeat\n");

#ifndef __aarch64__
    kprintf("\n  --- x86 Debug ---\n");
    kprintf("  rdmsr / wrmsr <msr> [v]   Read/write MSR\n");
    kprintf("  cpuid [leaf]              Raw CPUID dump\n");
    kprintf("  ioread / iowrite <p> [v]  I/O port access\n");
#endif

    kprintf("\n  --- Version Control ---\n");
    kprintf("  git init|add|commit|log|status|branch|diff\n");

    kprintf("\n  --- Package Manager ---\n");
    kprintf("  pkg install|search|list|info|remove|update|verify\n");

    kprintf("\n  --- Security ---\n");
    kprintf("  sandbox create|destroy    Sandbox management\n");
    kprintf("  sshd start|stop|status    SSH server control\n");
    kprintf("  sshd keygen|sessions      Regenerate host key / list sessions\n");
    kprintf("  sshd disconnect <id>      Disconnect SSH session\n");
    kprintf("  users [add|del] <name>    User management\n");
    kprintf("  passwd <user> <pass>      Change password\n");
    kprintf("  firewall enable|disable   Firewall control\n");
    kprintf("  firewall allow|deny <proto> <port>  Add rule\n");
    kprintf("  firewall rules|stats|flush          View/manage rules\n");
    kprintf("  audit [n|verify]          Security audit log\n");
    kprintf("  keystore [list|gen|del]   Cryptographic key store\n");
    kprintf("  integrity                 Verify system integrity\n");
    kprintf("  sec-init                  Initialize security framework\n");
    kprintf("  sha256 <string>           Compute SHA-256 hash\n");

    kprintf("\n  --- Subsystems ---\n");
    kprintf("  run <script.pseudo>       Execute script\n");
    kprintf("  ota / flash               OTA update\n");
    kprintf("  panic                     Test kernel panic\n");

    kprintf("\n  Line Editing: Up/Down=history, Left/Right=cursor,\n");
    kprintf("    Tab=complete, Ctrl+A/E=home/end, Ctrl+K/U/W=kill,\n");
    kprintf("    Ctrl+L=clear, Ctrl+C=cancel, Ctrl+T=transpose\n");
    kprintf("\n  Unrecognized input is JIT-compiled as Pseudocode.\n\n");
}

/* =============================================================================
 * COMMAND DISPATCH
 * =============================================================================*/

static int shell_exec_builtin(aishell_t *sh, int argc, char **argv)
{
    if (argc == 0) return 0;

    const char *cmd = argv[0];

    /* General */
    if (shell_strcmp(cmd, "help")     == 0) { shell_print_help(); return 0; }
    if (shell_strcmp(cmd, "status")   == 0) return cmd_status(sh, argc, argv);
    if (shell_strcmp(cmd, "sysinfo")  == 0) return cmd_sysinfo(sh, argc, argv);
    if (shell_strcmp(cmd, "uname")    == 0) return cmd_uname(sh, argc, argv);
    if (shell_strcmp(cmd, "uptime")   == 0) return cmd_uptime(sh, argc, argv);
    if (shell_strcmp(cmd, "clear")    == 0) return cmd_clear(sh, argc, argv);
    if (shell_strcmp(cmd, "reboot")   == 0) return cmd_reboot(sh, argc, argv);
    if (shell_strcmp(cmd, "whoami")   == 0) return cmd_whoami(sh, argc, argv);
    if (shell_strcmp(cmd, "hostname") == 0) return cmd_hostname(sh, argc, argv);
    if (shell_strcmp(cmd, "date")     == 0) return cmd_date(sh, argc, argv);
    if (shell_strcmp(cmd, "sleep")    == 0) return cmd_sleep(sh, argc, argv);
    if (shell_strcmp(cmd, "version")  == 0) return cmd_version(sh, argc, argv);
    if (shell_strcmp(cmd, "ver")      == 0) return cmd_version(sh, argc, argv);
    if (shell_strcmp(cmd, "about")    == 0) return cmd_about(sh, argc, argv);
    if (shell_strcmp(cmd, "logo")     == 0) return cmd_logo(sh, argc, argv);
    if (shell_strcmp(cmd, "halt")     == 0) return cmd_halt(sh, argc, argv);
    if (shell_strcmp(cmd, "panic")    == 0) return cmd_panic(sh, argc, argv);
    if (shell_strcmp(cmd, "true")     == 0) return cmd_true_cmd(sh, argc, argv);
    if (shell_strcmp(cmd, "false")    == 0) return cmd_false_cmd(sh, argc, argv);

    /* Filesystem */
    if (shell_strcmp(cmd, "ls")       == 0) return cmd_ls(sh, argc, argv);
    if (shell_strcmp(cmd, "dir")      == 0) return cmd_ls(sh, argc, argv);
    if (shell_strcmp(cmd, "cat")      == 0) return cmd_cat(sh, argc, argv);
    if (shell_strcmp(cmd, "stat")     == 0) return cmd_stat(sh, argc, argv);
    if (shell_strcmp(cmd, "mkdir")    == 0) return cmd_mkdir(sh, argc, argv);
    if (shell_strcmp(cmd, "touch")    == 0) return cmd_touch(sh, argc, argv);
    if (shell_strcmp(cmd, "write")    == 0) return cmd_write(sh, argc, argv);
    if (shell_strcmp(cmd, "rm")       == 0) return cmd_rm(sh, argc, argv);
    if (shell_strcmp(cmd, "pwd")      == 0) return cmd_pwd(sh, argc, argv);
    if (shell_strcmp(cmd, "cd")       == 0) return cmd_cd(sh, argc, argv);
    if (shell_strcmp(cmd, "cp")       == 0) return cmd_cp(sh, argc, argv);
    if (shell_strcmp(cmd, "mv")       == 0) return cmd_mv(sh, argc, argv);
    if (shell_strcmp(cmd, "head")     == 0) return cmd_head(sh, argc, argv);
    if (shell_strcmp(cmd, "tail")     == 0) return cmd_tail(sh, argc, argv);
    if (shell_strcmp(cmd, "wc")       == 0) return cmd_wc(sh, argc, argv);
    if (shell_strcmp(cmd, "grep")     == 0) return cmd_grep(sh, argc, argv);
    if (shell_strcmp(cmd, "find")     == 0) return cmd_find(sh, argc, argv);
    if (shell_strcmp(cmd, "df")       == 0) return cmd_df(sh, argc, argv);
    if (shell_strcmp(cmd, "du")       == 0) return cmd_du(sh, argc, argv);
    if (shell_strcmp(cmd, "tree")     == 0) return cmd_tree(sh, argc, argv);

    /* Hardware & System */
    if (shell_strcmp(cmd, "cpu")      == 0) return cmd_cpu(sh, argc, argv);
    if (shell_strcmp(cmd, "mem")      == 0) return cmd_mem(sh, argc, argv);
    if (shell_strcmp(cmd, "free")     == 0) return cmd_free_mem(sh, argc, argv);
    if (shell_strcmp(cmd, "top")      == 0) return cmd_top(sh, argc, argv);
    if (shell_strcmp(cmd, "disk")     == 0) return cmd_disk(sh, argc, argv);
    if (shell_strcmp(cmd, "smp")      == 0) return cmd_smp(sh, argc, argv);
    if (shell_strcmp(cmd, "net")      == 0) return cmd_net(sh, argc, argv);
    if (shell_strcmp(cmd, "ifconfig") == 0) return cmd_ifconfig(sh, argc, argv);
    if (shell_strcmp(cmd, "netstat")  == 0) return cmd_netstat(sh, argc, argv);
    if (shell_strcmp(cmd, "ping")     == 0) return cmd_ping(sh, argc, argv);
    if (shell_strcmp(cmd, "ps")       == 0) return cmd_ps(sh, argc, argv);
    if (shell_strcmp(cmd, "lspci")    == 0) return cmd_lspci(sh, argc, argv);
    if (shell_strcmp(cmd, "monitor")  == 0) return cmd_monitor(sh, argc, argv);
    if (shell_strcmp(cmd, "sched")    == 0) return cmd_sched(sh, argc, argv);
    if (shell_strcmp(cmd, "jit")      == 0) return cmd_jit(sh, argc, argv);
    if (shell_strcmp(cmd, "irq")      == 0) return cmd_irq(sh, argc, argv);
    if (shell_strcmp(cmd, "dmesg")    == 0) return cmd_dmesg(sh, argc, argv);

    /* AI / LLM */
    if (shell_strcmp(cmd, "llm")      == 0) return cmd_llm(sh, argc, argv);
    if (shell_strcmp(cmd, "ai")       == 0) return cmd_llm(sh, argc, argv);
    if (shell_strcmp(cmd, "reset")    == 0) return cmd_reset(sh, argc, argv);
    if (shell_strcmp(cmd, "infer")    == 0) return cmd_infer(sh, argc, argv);
    if (shell_strcmp(cmd, "train")    == 0) return cmd_train(sh, argc, argv);
    if (shell_strcmp(cmd, "deploy")   == 0) return cmd_deploy(sh, argc, argv);
    if (shell_strcmp(cmd, "serve")    == 0) return cmd_serve(sh, argc, argv);
    if (shell_strcmp(cmd, "api")      == 0) return cmd_serve(sh, argc, argv);

    /* Model Management */
    if (shell_strcmp(cmd, "model")    == 0) return cmd_model(sh, argc, argv);

    /* Tensor Operations */
    if (shell_strcmp(cmd, "tensor")   == 0) return cmd_tensor(sh, argc, argv);

    /* Developer Tools */
    if (shell_strcmp(cmd, "hexdump")  == 0) return cmd_hexdump(sh, argc, argv);
    if (shell_strcmp(cmd, "xxd")      == 0) return cmd_hexdump(sh, argc, argv);
    if (shell_strcmp(cmd, "peek")     == 0) return cmd_peek(sh, argc, argv);
    if (shell_strcmp(cmd, "poke")     == 0) return cmd_poke(sh, argc, argv);
    if (shell_strcmp(cmd, "strings")  == 0) return cmd_strings(sh, argc, argv);
    if (shell_strcmp(cmd, "crc32")    == 0) return cmd_crc32(sh, argc, argv);
    if (shell_strcmp(cmd, "calc")     == 0) return cmd_calc(sh, argc, argv);
    if (shell_strcmp(cmd, "time")     == 0) return cmd_time(sh, argc, argv);
    if (shell_strcmp(cmd, "repeat")   == 0) return cmd_repeat(sh, argc, argv);
    if (shell_strcmp(cmd, "echo")     == 0) return cmd_echo(sh, argc, argv);
    if (shell_strcmp(cmd, "history")  == 0) return cmd_history(sh, argc, argv);
    if (shell_strcmp(cmd, "selftest") == 0) return cmd_selftest(sh, argc, argv);
    if (shell_strcmp(cmd, "test")     == 0) return cmd_selftest(sh, argc, argv);
    if (shell_strcmp(cmd, "yes")      == 0) return cmd_yes(sh, argc, argv);
    if (shell_strcmp(cmd, "seq")      == 0) return cmd_seq(sh, argc, argv);
    if (shell_strcmp(cmd, "rand")     == 0) return cmd_rand(sh, argc, argv);

    /* Shell features */
    if (shell_strcmp(cmd, "alias")    == 0) return cmd_alias(sh, argc, argv);
    if (shell_strcmp(cmd, "unalias")  == 0) return cmd_unalias(sh, argc, argv);
    if (shell_strcmp(cmd, "set")      == 0) return cmd_set(sh, argc, argv);
    if (shell_strcmp(cmd, "env")      == 0) return cmd_env(sh, argc, argv);
    if (shell_strcmp(cmd, "export")   == 0) return cmd_set(sh, argc, argv);

#ifndef __aarch64__
    /* x86 debug */
    if (shell_strcmp(cmd, "rdmsr")    == 0) return cmd_rdmsr(sh, argc, argv);
    if (shell_strcmp(cmd, "wrmsr")    == 0) return cmd_wrmsr(sh, argc, argv);
    if (shell_strcmp(cmd, "cpuid")    == 0) return cmd_cpuid_cmd(sh, argc, argv);
    if (shell_strcmp(cmd, "ioread")   == 0) return cmd_ioread(sh, argc, argv);
    if (shell_strcmp(cmd, "iowrite")  == 0) return cmd_iowrite(sh, argc, argv);
#endif

    /* Benchmarks & Demos */
    if (shell_strcmp(cmd, "bench")    == 0) return cmd_bench(sh, argc, argv);
    if (shell_strcmp(cmd, "demo")     == 0) return cmd_demo(sh, argc, argv);

    /* Version Control */
    if (shell_strcmp(cmd, "git")      == 0) return cmd_git(sh, argc, argv);

    /* Package Manager */
    if (shell_strcmp(cmd, "pkg")      == 0) return cmd_pkg(sh, argc, argv);

    /* Security */
    if (shell_strcmp(cmd, "sandbox")  == 0) return cmd_sandbox(sh, argc, argv);
    if (shell_strcmp(cmd, "sshd")     == 0) return cmd_sshd(sh, argc, argv);
    if (shell_strcmp(cmd, "users")    == 0) return cmd_users(sh, argc, argv);
    if (shell_strcmp(cmd, "passwd")   == 0) return cmd_passwd(sh, argc, argv);
    if (shell_strcmp(cmd, "firewall") == 0) return cmd_firewall(sh, argc, argv);
    if (shell_strcmp(cmd, "fw")       == 0) return cmd_firewall(sh, argc, argv);
    if (shell_strcmp(cmd, "audit")    == 0) return cmd_audit(sh, argc, argv);
    if (shell_strcmp(cmd, "keystore") == 0) return cmd_keystore(sh, argc, argv);
    if (shell_strcmp(cmd, "integrity")== 0) return cmd_integrity(sh, argc, argv);
    if (shell_strcmp(cmd, "sec-init") == 0) return cmd_sec_init(sh, argc, argv);
    if (shell_strcmp(cmd, "sha256")   == 0) return cmd_sha256(sh, argc, argv);

    /* Subsystems */
    if (shell_strcmp(cmd, "run")      == 0) return cmd_run(sh, argc, argv);
    if (shell_strcmp(cmd, "ota")      == 0) return cmd_ota(sh, argc, argv);
    if (shell_strcmp(cmd, "flash")    == 0) return cmd_flash(sh, argc, argv);

    return -1; /* Not a builtin */
}

/* =============================================================================
 * BANNER
 * =============================================================================*/

static void shell_print_banner(void)
{
    kprintf("\n");
    kprintf("                         :-==-:                         \n");
    kprintf("                      :-========-:                      \n");
    kprintf("                  .:================:.                  \n");
    kprintf("               .:=======--::::--=======:.               \n");
    kprintf("            .-=======--::::::::::--=======-.            \n");
    kprintf("         .-=======--::::::::::::::::--=======-.         \n");
    kprintf("      :-=======--::::::::::::::::::::::--=======-:      \n");
    kprintf("     .:-====--::::::::::::::::::::::::::::--=====--     \n");
    kprintf("     ....:--::::::::::::::::::::::::::::::::::-::::     \n");
    kprintf("     .....===--:::::::::::::::::::::::::::....:::::     \n");
    kprintf("     .....======--:::::::::::::::::::::.......:::::     \n");
    kprintf("     .....==========--::::::::::::::..........:::::     \n");
    kprintf("     .....=============--::::::::.............:::::     \n");
    kprintf("     .....================--:.................:::::     \n");
    kprintf("     .....==================..................::::-     \n");
    kprintf("     .....==================..................::::-     \n");
    kprintf("     .....==================..................::::-     \n");
    kprintf("     .....==================..................::::-     \n");
    kprintf("     .....==================..................::::-     \n");
    kprintf("     .....==================..................::::-     \n");
    kprintf("     ......:-===============................::::::-     \n");
    kprintf("      ........:-============.............:::::::-:      \n");
    kprintf("         .........:-========.........:::::::--:         \n");
    kprintf("            .........:-=====......:::::::-:.            \n");
    kprintf("               .........:-==...:::::::-:.               \n");
    kprintf("                   .........:::::::-:.                  \n");
    kprintf("                      ......::::-:                      \n");
    kprintf("                         ...--:                         \n");
    kprintf("\n");
    kprintf("  TensorOS v0.1.0 - AI-First Operating System\n");
    kprintf("  %d CPU(s), %lu MB RAM", kstate.cpu_count,
            kstate.memory_total_bytes / (1024*1024));
    if (llm_is_loaded())
        kprintf(", LLM: %s", llm_model_name());
    kprintf("\n");
    kprintf("  Type 'help' for commands, 'llm <prompt>' to chat with AI.\n\n");
}

/* =============================================================================
 * MAIN SHELL ENTRY
 * =============================================================================*/

void aishell_init(aishell_t *sh)
{
    kmemset(sh, 0, sizeof(*sh));
    shell_strncpy(sh->prompt, "tensor> ", SHELL_PROMPT_MAX);
    sh->running = true;
    sh->interactive = true;
    sh->session_start_ticks = kstate.uptime_ticks;

    sh->runtime = (pseudo_runtime_t *)kmalloc(sizeof(pseudo_runtime_t));
    if (sh->runtime)
        kmemset(sh->runtime, 0, sizeof(*sh->runtime));
}

void aishell_run(aishell_t *sh)
{
    shell_print_banner();

    char line[SHELL_MAX_LINE];
    char *argv[SHELL_MAX_ARGS];
    int argc;

    while (sh->running) {
        kprintf("%s", sh->prompt);

        int len = shell_read_line(sh, line, SHELL_MAX_LINE);
        if (len == 0) continue;

        history_add(&sh->history, line);

        if (shell_strcmp(line, "exit") == 0 || shell_strcmp(line, "quit") == 0) {
            kprintf("Shutting down TensorOS...\n");
            sh->running = false;
            break;
        }

        shell_parse_line(line, &argc, argv);

        /* Alias expansion: if first token matches an alias, substitute */
        if (argc > 0) {
            for (int i = 0; i < sh->alias_count; i++) {
                if (shell_strcmp(argv[0], sh->aliases[i].name) == 0) {
                    static char expanded[SHELL_MAX_LINE];
                    int epos = 0;
                    for (const char *p = sh->aliases[i].command; *p && epos < SHELL_MAX_LINE - 2; p++)
                        expanded[epos++] = *p;
                    for (int j = 1; j < argc && epos < SHELL_MAX_LINE - 2; j++) {
                        expanded[epos++] = ' ';
                        for (const char *p = argv[j]; *p && epos < SHELL_MAX_LINE - 2; p++)
                            expanded[epos++] = *p;
                    }
                    expanded[epos] = '\0';
                    shell_parse_line(expanded, &argc, argv);
                    break;
                }
            }
        }

        int r = shell_exec_builtin(sh, argc, argv);
        sh->last_exit_code = r;

        if (r == -1) {
            /* Not a builtin — try Pseudocode JIT */
            kprintf("[JIT] Compiling: %s\n", line);
            if (sh->runtime)
                pseudo_exec_string(sh->runtime, line);
        }

        sh->commands_executed++;
    }
}

void aishell_main(void)
{
    static aishell_t shell;  /* Static to avoid 4KB+ on the 64KB stack */
    aishell_init(&shell);
    aishell_run(&shell);
}
