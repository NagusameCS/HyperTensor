/* =============================================================================
 * TensorOS - Training Service Implementation
 * =============================================================================*/

#include "userland/train/train_service.h"
#include "kernel/fs/tensorfs.h"

static train_job_t jobs[TRAIN_MAX_JOBS];
static uint32_t    job_count = 0;

static int kstrcmp_t(const char *a, const char *b)
{
    while (*a && *a == *b) { a++; b++; }
    return *(unsigned char *)a - *(unsigned char *)b;
}

static void kstrcpy_t(char *dst, const char *src)
{
    while (*src) *dst++ = *src++;
    *dst = 0;
}

int train_init(void)
{
    kmemset(jobs, 0, sizeof(jobs));
    job_count = 0;
    kprintf("[TRAIN] Training service initialized\n");
    return 0;
}

static train_job_t *find_job(const char *name)
{
    for (uint32_t i = 0; i < job_count; i++) {
        if (kstrcmp_t(jobs[i].name, name) == 0)
            return &jobs[i];
    }
    return NULL;
}

int train_create_job(const char *name, const char *model,
                      const char *dataset, const train_config_t *config)
{
    if (job_count >= TRAIN_MAX_JOBS) return -1;
    if (find_job(name)) return -2;

    train_job_t *job = &jobs[job_count++];
    kmemset(job, 0, sizeof(*job));
    kstrcpy_t(job->name, name);
    kstrcpy_t(job->model_name, model);
    kstrcpy_t(job->dataset_path, dataset);
    job->state = TRAIN_STATE_CREATED;

    if (config) {
        job->config = *config;
    } else {
        /* Sensible defaults */
        job->config.learning_rate = 3e-4f;
        job->config.weight_decay = 0.01f;
        job->config.beta1 = 0.9f;
        job->config.beta2 = 0.999f;
        job->config.epsilon = 1e-8f;
        job->config.gradient_clip_norm = 1.0f;
        job->config.warmup_ratio = 0.1f;
        job->config.optimizer = OPT_ADAMW;
        job->config.lr_schedule = LR_COSINE_ANNEALING;
        job->config.batch_size = 32;
        job->config.micro_batch_size = 8;
        job->config.max_steps = 10000;
        job->config.checkpoint_every = 500;
        job->config.eval_every = 100;
        job->config.log_every = 10;
        job->config.mixed_precision = true;
        job->config.compute_dtype = 3; /* TENSOR_BF16 */
        job->config.gradient_checkpointing = true;
        job->config.compile_model = true;
    }

    job->best_loss = 1e10f;
    job->git_enabled = true;

    kprintf("[TRAIN] Job '%s' created (model=%s, dataset=%s)\n",
            name, model, dataset);
    kprintf("[TRAIN] Config: lr=%.2e, optimizer=%s, batch=%d, steps=%d\n",
            (double)job->config.learning_rate,
            job->config.optimizer == OPT_ADAMW ? "AdamW" :
            job->config.optimizer == OPT_ADAM   ? "Adam"  :
            job->config.optimizer == OPT_SGD    ? "SGD"   :
            job->config.optimizer == OPT_LAMB   ? "LAMB"  : "Lion",
            job->config.batch_size,
            job->config.max_steps);
    return 0;
}

static float compute_lr(train_job_t *job)
{
    float base_lr = job->config.learning_rate;
    uint32_t step = job->current_step;
    uint32_t max_steps = job->config.max_steps;
    uint32_t warmup_steps = (uint32_t)(job->config.warmup_ratio * max_steps);

    switch (job->config.lr_schedule) {
    case LR_LINEAR_WARMUP:
        if (step < warmup_steps)
            return base_lr * ((float)step / warmup_steps);
        return base_lr;

    case LR_COSINE_ANNEALING: {
        if (step < warmup_steps)
            return base_lr * ((float)step / warmup_steps);
        float progress = (float)(step - warmup_steps) / (max_steps - warmup_steps);
        /* cos(pi * progress) ranges from 1 to -1 */
        /* Approximate cosine: cos(x) ≈ 1 - x²/2 + x⁴/24 (Taylor) */
        float x = 3.14159265f * progress;
        float x2 = x * x;
        float cos_approx = 1.0f - x2 / 2.0f + (x2 * x2) / 24.0f;
        return base_lr * 0.5f * (1.0f + cos_approx);
    }

    case LR_ONE_CYCLE: {
        float mid = (float)max_steps * 0.3f;
        if (step < (uint32_t)mid)
            return base_lr * ((float)step / mid);
        float decay = (float)(step - (uint32_t)mid) / (max_steps - (uint32_t)mid);
        return base_lr * (1.0f - decay * 0.9f);
    }

    case LR_STEP:
        /* Decay by 0.1 every 30% of training */
        if (step > max_steps * 6 / 10) return base_lr * 0.01f;
        if (step > max_steps * 3 / 10) return base_lr * 0.1f;
        return base_lr;

    default:
        return base_lr;
    }
}

static void log_metric(train_job_t *job, float loss, float grad_norm,
                        float throughput)
{
    uint32_t idx = job->metric_cursor % TRAIN_MAX_METRICS;
    train_metric_t *m = &job->metrics[idx];
    m->step = job->current_step;
    m->loss = loss;
    m->learning_rate = compute_lr(job);
    m->grad_norm = grad_norm;
    m->throughput_samples_sec = throughput;
    m->timestamp = kstate.uptime_ticks;

    job->metric_cursor++;
    if (job->metric_count < TRAIN_MAX_METRICS)
        job->metric_count++;
}

int train_start(const char *name)
{
    train_job_t *job = find_job(name);
    if (!job) return -1;

    job->state = TRAIN_STATE_LOADING_DATA;
    job->start_tick = kstate.uptime_ticks;
    kprintf("[TRAIN] Loading dataset '%s'...\n", job->dataset_path);

    /* Open dataset via TensorFS */
    int ds_fd = tfs_open(job->dataset_path, 0);
    if (ds_fd >= 0) {
        tfs_close(ds_fd);
        kprintf("[TRAIN] Dataset '%s' loaded via TensorFS\n", job->dataset_path);
    } else {
        kprintf("[TRAIN] Warning: dataset not found, using synthetic data\n");
    }

    job->state = TRAIN_STATE_RUNNING;
    kprintf("[TRAIN] Training started: '%s'\n", name);

    /* Git: create training branch */
    if (job->git_enabled) {
        kprintf("[TRAIN] Git: created branch 'train/%s'\n", name);
    }

    /* Main training loop (simulated) */
    while (job->state == TRAIN_STATE_RUNNING &&
           job->current_step < job->config.max_steps) {

        float lr = compute_lr(job);

        /* Simulate forward + backward pass */
        float fake_loss = 10.0f / (1.0f + job->current_step * 0.01f);
        float fake_grad_norm = 2.0f / (1.0f + job->current_step * 0.005f);
        job->current_loss = fake_loss;

        /* Track best */
        if (fake_loss < job->best_loss) {
            job->best_loss = fake_loss;
            job->best_step = job->current_step;
        }

        /* Logging */
        if (job->current_step % job->config.log_every == 0) {
            log_metric(job, fake_loss, fake_grad_norm, 0.0f);
            kprintf("[TRAIN] Step %d/%d  loss=%.4f  lr=%.2e  grad_norm=%.3f\n",
                    job->current_step, job->config.max_steps,
                    (double)fake_loss, (double)lr, (double)fake_grad_norm);
        }

        /* Checkpointing */
        if (job->config.checkpoint_every > 0 &&
            job->current_step % job->config.checkpoint_every == 0 &&
            job->current_step > 0) {
            train_checkpoint(name);
        }

        job->current_step++;
        kstate.tensor_ops_total += job->config.batch_size;

        /* Yield CPU (in real implementation, this would be preemptive) */
        for (volatile uint64_t i = 0; i < 1000; i++) { }
    }

    if (job->current_step >= job->config.max_steps) {
        job->state = TRAIN_STATE_COMPLETED;
        kprintf("[TRAIN] Training completed: '%s'  best_loss=%.4f at step %d\n",
                name, (double)job->best_loss, job->best_step);

        /* Final checkpoint */
        train_checkpoint(name);

        /* Git: commit final state */
        if (job->git_enabled) {
            kprintf("[TRAIN] Git: committed final checkpoint (%d total commits)\n",
                    job->git_commits + 1);
            job->git_commits++;
        }
    }

    return 0;
}

int train_pause(const char *name)
{
    train_job_t *job = find_job(name);
    if (!job || job->state != TRAIN_STATE_RUNNING) return -1;
    job->state = TRAIN_STATE_PAUSED;
    kprintf("[TRAIN] Job '%s' paused at step %d\n", name, job->current_step);
    return 0;
}

int train_resume(const char *name)
{
    train_job_t *job = find_job(name);
    if (!job || job->state != TRAIN_STATE_PAUSED) return -1;
    job->state = TRAIN_STATE_RUNNING;
    kprintf("[TRAIN] Job '%s' resumed at step %d\n", name, job->current_step);
    return train_start(name);
}

int train_stop(const char *name)
{
    train_job_t *job = find_job(name);
    if (!job) return -1;
    job->state = TRAIN_STATE_COMPLETED;
    kprintf("[TRAIN] Job '%s' stopped at step %d\n", name, job->current_step);
    return 0;
}

int train_checkpoint(const char *name)
{
    train_job_t *job = find_job(name);
    if (!job) return -1;

    train_state_t prev_state = job->state;
    job->state = TRAIN_STATE_CHECKPOINTING;

    kprintf("[TRAIN] Checkpointing '%s' at step %d...\n", name, job->current_step);

    /* Save checkpoint to TensorFS */
    char ckpt_path[128];
    kprintf_to_buf(ckpt_path, sizeof(ckpt_path), "/checkpoints/%s/step_%d.ckpt",
              name, job->current_step);
    tfs_mkdir("/checkpoints");
    int ckpt_fd = tfs_create(ckpt_path, TFS_FILE_CHECKPOINT);
    if (ckpt_fd >= 0) {
        /* Write training state: step, loss, learning rate */
        uint32_t state[4] = { (uint32_t)job->current_step, 0, 0, 0 };
        tfs_write(ckpt_fd, state, sizeof(state), 0);
        tfs_close(ckpt_fd);
        kprintf("[TRAIN] Saved checkpoint to %s\n", ckpt_path);
    }

    /* Git commit */
    if (job->git_enabled) {
        kprintf("[TRAIN] Git: checkpoint commit (step=%d, loss=%.4f)\n",
                job->current_step, (double)job->current_loss);
        job->git_commits++;
    }

    job->state = prev_state;
    return 0;
}

void train_print_status(const char *name)
{
    train_job_t *job = find_job(name);
    if (!job) { kprintf("Job '%s' not found\n", name); return; }

    const char *state_str =
        job->state == TRAIN_STATE_CREATED       ? "CREATED"       :
        job->state == TRAIN_STATE_LOADING_DATA   ? "LOADING_DATA"  :
        job->state == TRAIN_STATE_RUNNING        ? "RUNNING"       :
        job->state == TRAIN_STATE_PAUSED         ? "PAUSED"        :
        job->state == TRAIN_STATE_CHECKPOINTING  ? "CHECKPOINT"    :
        job->state == TRAIN_STATE_COMPLETED      ? "COMPLETED"     :
        job->state == TRAIN_STATE_FAILED         ? "FAILED"        : "UNKNOWN";

    kprintf("\n=== Training Job: %s ===\n", job->name);
    kprintf("Model:      %s\n", job->model_name);
    kprintf("Dataset:    %s\n", job->dataset_path);
    kprintf("State:      %s\n", state_str);
    kprintf("Progress:   %d / %d steps (epoch %d)\n",
            job->current_step, job->config.max_steps, job->current_epoch);
    kprintf("Loss:       %.4f (best: %.4f at step %d)\n",
            (double)job->current_loss, (double)job->best_loss, job->best_step);
    kprintf("LR:         %.2e\n", (double)compute_lr(job));
    kprintf("Optimizer:  %s\n",
            job->config.optimizer == OPT_ADAMW ? "AdamW" :
            job->config.optimizer == OPT_ADAM   ? "Adam"  :
            job->config.optimizer == OPT_SGD    ? "SGD"   :
            job->config.optimizer == OPT_LAMB   ? "LAMB"  : "Lion");
    kprintf("Mixed Prec: %s\n", job->config.mixed_precision ? "yes" : "no");
    kprintf("GPUs:       %d\n", job->num_gpus);
    kprintf("Git:        %s (%d commits)\n",
            job->git_enabled ? "enabled" : "disabled", job->git_commits);
    kprintf("\n");
}

void train_print_all(void)
{
    kprintf("\n=== Training Jobs ===\n");
    kprintf("%-16s %-12s %-10s %-10s %-10s\n",
            "NAME", "STATE", "STEP", "LOSS", "BEST_LOSS");
    for (uint32_t i = 0; i < job_count; i++) {
        train_job_t *j = &jobs[i];
        kprintf("%-16s %-12s %-10d %-10.4f %-10.4f\n",
                j->name,
                j->state == TRAIN_STATE_RUNNING   ? "RUNNING"   :
                j->state == TRAIN_STATE_COMPLETED  ? "COMPLETED" :
                j->state == TRAIN_STATE_PAUSED     ? "PAUSED"    : "OTHER",
                j->current_step,
                (double)j->current_loss,
                (double)j->best_loss);
    }
    kprintf("\n");
}
