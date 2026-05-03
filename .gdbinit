# TensorOS GDB configuration
# Usage: gdb -x .gdbinit build/tensoros.bin

# Connect to QEMU GDB stub
target remote localhost:1234

# Load kernel symbols
symbol-file build/tensoros.bin

# Set architecture
set architecture i386:x86-64

# Don't ask for confirmations
set confirm off

# Display source on stop
set listsize 20

# Useful breakpoints
break kernel_main
break tensor_sched_init
break tensor_mm_init
break aishell_main

# Custom commands
define tensor-state
    print kstate
end
document tensor-state
    Print the global kernel state (MEU count, GPU count, tensor ops, etc.)
end

define meu-list
    set $i = 0
    while $i < kstate.meu_count
        printf "MEU #%d: %s (state=%d, priority=%d)\n", \
            kstate.meus[$i].id, kstate.meus[$i].name, \
            kstate.meus[$i].state, kstate.meus[$i].priority
        set $i = $i + 1
    end
end
document meu-list
    List all Model Execution Units with their state
end

define gpu-info
    set $i = 0
    while $i < kstate.gpu_count
        printf "GPU #%d: vendor=0x%04x device=0x%04x vram=%llu MB\n", \
            $i, 0, 0, 0
        set $i = $i + 1
    end
end
document gpu-info
    Show detected GPU information
end

# Start execution
echo \n
echo ============================================\n
echo   TensorOS GDB Session\n
echo   Breakpoint set at kernel_main\n
echo   Type 'continue' to start booting\n
echo   Custom commands: tensor-state, meu-list, gpu-info\n
echo ============================================\n
echo \n
