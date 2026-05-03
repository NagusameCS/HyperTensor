# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
# ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
# ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
# ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
# ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
# ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
# :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
# :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
# ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
# :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
# ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
# ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
# :::::::::................................:@@@@@@@@@@%:...............................::::::
# ::::::::..................................*@@@@@@@@@-................................::::::::
# ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
# :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
# :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
# :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
# :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
# :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
# :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
# :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
# :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
# ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
# ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
# :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
# ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
# ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
# :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
# ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
# :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
# :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
# ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
# ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
# :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
# ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
# ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
# ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# ==============================================================================
# TensorOS — Master Makefile
# ==============================================================================
# Build targets:
#   make              Build kernel image (tensoros.bin)
#   make iso          Build bootable ISO (tensoros.iso)
#   make run          Build and run in QEMU
#   make debug        Build with debug symbols and run in QEMU with GDB stub
#   make clean        Remove build artifacts
#
# Requirements:
#   - x86_64-elf cross-compiler (or system gcc targeting x86_64)
#   - nasm assembler
#   - grub-mkrescue + xorriso (for ISO)
#   - qemu-system-x86_64
# ==============================================================================

# ---- Toolchain ----
CC       := x86_64-elf-gcc
AS       := nasm
LD       := x86_64-elf-ld
OBJCOPY  := x86_64-elf-objcopy

# If cross-compiler not found, fall back to system gcc
ifeq ($(shell which $(CC) 2>/dev/null),)
  CC     := gcc
  LD     := ld
  OBJCOPY := objcopy
endif

# ---- Flags ----
CFLAGS   := -ffreestanding -nostdlib -nostdinc -fno-builtin -fno-stack-protector \
            -mno-red-zone -mno-mmx -mno-sse -mno-sse2 -Wall -Wextra -O2 \
            -mcmodel=kernel -fno-pic -fno-pie
ASFLAGS  := -f elf64
LDFLAGS  := -T boot/linker.ld -nostdlib -z max-page-size=0x1000

DEBUG_CFLAGS := $(CFLAGS) -g -DTENSOROS_DEBUG -O0

# ---- Directories ----
BUILD    := build
ISO_DIR  := $(BUILD)/isodir

# ---- Source files ----
ASM_SRC  := boot/boot.asm

C_SRC    := kernel/core/main.c \
            kernel/sched/tensor_sched.c \
            kernel/mm/tensor_mm.c \
            kernel/drivers/gpu/gpu.c \
            kernel/drivers/tpu/tpu.c \
            kernel/fs/git.c \
            kernel/fs/tensorfs.c \
            kernel/security/sandbox.c \
            kernel/ipc/tensor_ipc.c \
            virt/virt.c \
            runtime/pseudocode/pseudocode_jit.c \
            runtime/tensor/tensor_engine.c \
            pkg/modelpkg.c \
            userland/shell/aishell.c \
            userland/monitor/tensor_monitor.c \
            userland/deploy/deploy_service.c \
            userland/train/train_service.c

# ---- Object files ----
ASM_OBJ  := $(patsubst %.asm,$(BUILD)/%.o,$(ASM_SRC))
C_OBJ    := $(patsubst %.c,$(BUILD)/%.o,$(C_SRC))
OBJ      := $(ASM_OBJ) $(C_OBJ)

# ---- Targets ----
KERNEL   := $(BUILD)/tensoros.bin
ISO      := $(BUILD)/tensoros.iso

.PHONY: all iso run debug clean dirs

all: dirs $(KERNEL)
	@echo "================================================================"
	@echo "  TensorOS kernel built: $(KERNEL)"
	@echo "  Size: $$(stat -c%s $(KERNEL) 2>/dev/null || stat -f%z $(KERNEL)) bytes"
	@echo "================================================================"

# Create build directory tree
dirs:
	@mkdir -p $(BUILD)/boot
	@mkdir -p $(BUILD)/kernel/core
	@mkdir -p $(BUILD)/kernel/sched
	@mkdir -p $(BUILD)/kernel/mm
	@mkdir -p $(BUILD)/kernel/drivers/gpu
	@mkdir -p $(BUILD)/kernel/drivers/tpu
	@mkdir -p $(BUILD)/kernel/fs
	@mkdir -p $(BUILD)/kernel/security
	@mkdir -p $(BUILD)/kernel/ipc
	@mkdir -p $(BUILD)/virt
	@mkdir -p $(BUILD)/runtime/pseudocode
	@mkdir -p $(BUILD)/runtime/tensor
	@mkdir -p $(BUILD)/pkg
	@mkdir -p $(BUILD)/userland/shell
	@mkdir -p $(BUILD)/userland/monitor
	@mkdir -p $(BUILD)/userland/deploy
	@mkdir -p $(BUILD)/userland/train

# Link kernel
$(KERNEL): $(OBJ)
	@echo "[LD] Linking kernel..."
	$(LD) $(LDFLAGS) -o $@ $^
	@echo "[OK] Kernel linked: $@"

# Assemble .asm → .o
$(BUILD)/%.o: %.asm
	@echo "[AS] $<"
	$(AS) $(ASFLAGS) -o $@ $<

# Compile .c → .o
$(BUILD)/%.o: %.c
	@echo "[CC] $<"
	$(CC) $(CFLAGS) -c -o $@ $<

# ---- ISO target (Multiboot2 ISO via GRUB) ----
iso: all
	@echo "[ISO] Building bootable ISO..."
	@mkdir -p $(ISO_DIR)/boot/grub
	@cp $(KERNEL) $(ISO_DIR)/boot/tensoros.bin
	@echo 'set timeout=0'                        >  $(ISO_DIR)/boot/grub/grub.cfg
	@echo 'set default=0'                        >> $(ISO_DIR)/boot/grub/grub.cfg
	@echo ''                                      >> $(ISO_DIR)/boot/grub/grub.cfg
	@echo 'menuentry "TensorOS" {'                >> $(ISO_DIR)/boot/grub/grub.cfg
	@echo '    multiboot2 /boot/tensoros.bin'     >> $(ISO_DIR)/boot/grub/grub.cfg
	@echo '    boot'                              >> $(ISO_DIR)/boot/grub/grub.cfg
	@echo '}'                                     >> $(ISO_DIR)/boot/grub/grub.cfg
	grub-mkrescue -o $(ISO) $(ISO_DIR)
	@echo "[OK] ISO built: $(ISO)"

# ---- QEMU targets ----
QEMU     := qemu-system-x86_64
QEMU_MEM := 4G

# Common QEMU flags
QEMU_FLAGS := -m $(QEMU_MEM) \
              -smp 4 \
              -kernel $(KERNEL) \
              -serial stdio \
              -no-reboot \
              -no-shutdown \
              -d int,cpu_reset \
              -D $(BUILD)/qemu.log

# GPU: emulate virtio-gpu (QEMU doesn't emulate real NVIDIA GPUs)
QEMU_GPU := -device virtio-gpu-pci

# Network
QEMU_NET := -netdev user,id=net0,hostfwd=tcp::8080-:8080 \
            -device virtio-net-pci,netdev=net0

# Optional KVM acceleration (Linux only)
QEMU_KVM := $(shell [ -e /dev/kvm ] && echo "-enable-kvm -cpu host" || echo "-cpu qemu64,+sse2")

run: all
	@echo "[QEMU] Launching TensorOS..."
	$(QEMU) $(QEMU_FLAGS) $(QEMU_GPU) $(QEMU_NET) $(QEMU_KVM)

# Debug: add GDB stub, don't start CPU until GDB connects
debug: CFLAGS := $(DEBUG_CFLAGS)
debug: clean all
	@echo "[QEMU] Launching TensorOS in DEBUG mode (waiting for GDB on :1234)..."
	$(QEMU) $(QEMU_FLAGS) $(QEMU_GPU) $(QEMU_NET) $(QEMU_KVM) \
	        -s -S

# ---- ISO run ----
run-iso: iso
	@echo "[QEMU] Booting from ISO..."
	$(QEMU) -m $(QEMU_MEM) -smp 4 -cdrom $(ISO) -serial stdio \
	        $(QEMU_GPU) $(QEMU_NET) $(QEMU_KVM)

# ---- Clean ----
clean:
	@echo "[CLEAN] Removing build artifacts..."
	rm -rf $(BUILD)
	@echo "[OK] Clean."
