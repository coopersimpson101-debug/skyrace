/*
 * skyrace_sitl.c - software-in-the-loop shim around the REAL skyforge-fc code.
 *
 * Built against a skyforge-fc checkout (see scripts/build_sitl.py): it
 * compiles every skyforge-fc src/core C file and #includes its sim/quad_sim.c, so the
 * rigid-body model, the sensor model, the AHRS, the rate PIDs, the mixer,
 * arming and failsafe are all skyforge-fc's own C code, unmodified.
 *
 * What this file adds: a byte-level receiver UART. Instead of quad_sim's
 * periodic sim_set_rc(), RC commands only reach the FC when a complete,
 * CRC-valid CRSF RC_CHANNELS frame has been parsed by skyforge-fc's
 * crsf_parse_byte() and mapped by fc_rc_from_crsf() - exactly the path in
 * skyforge-fc src/app/main.c (task_rx). Telemetry (ATTITUDE + FLIGHT_MODE)
 * is produced at 10 Hz the same way task_telemetry() does.
 */
#include "quad_sim.c" /* static physics()/sense() become visible here */

#include <string.h>

#include "crsf.h"
#include "fmt.h"

#ifdef _WIN32
#define SITL_API __declspec(dllexport)
#else
#define SITL_API __attribute__((visibility("default")))
#endif

#define SITL_TX_CAP 4096

typedef struct {
    quad_sim_t sim;
    crsf_parser_t rx;
    uint8_t tx[SITL_TX_CAP];
    int ntx;
    uint32_t next_telem_us;
    uint32_t rc_frames;
    float thrust;
} sitl_t;

SITL_API sitl_t *sitl_create(void)
{
    sitl_t *s = (sitl_t *)calloc(1, sizeof(sitl_t));
    return s;
}

SITL_API void sitl_destroy(sitl_t *s)
{
    free(s);
}

SITL_API void sitl_reset(sitl_t *s, uint32_t seed, float x, float y, float z, float yaw)
{
    sim_reset(&s->sim, seed);
    s->sim.pos = v3(x, y, z);
    s->sim.q = q_from_euler(0.0f, 0.0f, yaw);
    s->sim.on_ground = z >= 0.0f;
    s->sim.link_up = false; /* RC only via the UART below */
    crsf_parser_init(&s->rx);
    s->ntx = 0;
    s->next_telem_us = s->sim.now_us + 100000u;
    s->rc_frames = 0;
}

SITL_API int sitl_set_param(sitl_t *s, const char *name, float value)
{
    return sim_set_param(&s->sim, name, value);
}

SITL_API void sitl_set_airframe(sitl_t *s, float mass, float thrust_max, float drag_lin)
{
    if (mass > 0.0f) s->sim.mass = mass;
    if (thrust_max > 0.0f) s->sim.thrust_max = thrust_max;
    if (drag_lin >= 0.0f) s->sim.drag_lin = drag_lin;
}

SITL_API void sitl_uart_write(sitl_t *s, const uint8_t *data, int n)
{
    for (int i = 0; i < n; i++) {
        if (crsf_parse_byte(&s->rx, data[i], s->sim.now_us) == CRSF_TYPE_RC_CHANNELS) {
            rc_command_t rc;
            fc_rc_from_crsf(&s->sim.cfg, s->rx.channels, &rc);
            fc_rc_update(&s->sim.fc, &rc, s->sim.now_us);
            s->sim.rc = rc;
            s->rx.rc_fresh = false;
            s->rc_frames++;
        }
    }
}

SITL_API int sitl_uart_read(sitl_t *s, uint8_t *out, int cap)
{
    int n = s->ntx < cap ? s->ntx : cap;
    memcpy(out, s->tx, (size_t)n);
    memmove(s->tx, s->tx + n, (size_t)(s->ntx - n));
    s->ntx -= n;
    return n;
}

static void tx_push(sitl_t *s, const uint8_t *f, int n)
{
    if (s->ntx + n <= SITL_TX_CAP) {
        memcpy(s->tx + s->ntx, f, (size_t)n);
        s->ntx += n;
    }
}

static void telemetry(sitl_t *s)
{
    /* same content as skyforge-fc src/app/main.c task_telemetry() */
    uint8_t frame[CRSF_MAX_FRAME];
    float r, p, y;
    fc_t *fc = &s->sim.fc;
    q_to_euler(fc->q, &r, &p, &y);
    tx_push(s, frame, crsf_build_attitude(frame, p, r, y));
    char mode[16] = {0};
    if (fc->fs != FS_IDLE) {
        fmt_append(mode, sizeof(mode), "!FS!");
    } else {
        fmt_append(mode, sizeof(mode), fc->mode == MODE_ANGLE ? "ANGL" : "ACRO");
        if (!fc->armed) fmt_append(mode, sizeof(mode), "*");
    }
    tx_push(s, frame, crsf_build_flight_mode(frame, mode));
}

SITL_API void sitl_step(sitl_t *s, int n)
{
    quad_sim_t *q = &s->sim;
    const float dt = 1.0f / (float)q->cfg.gyro_rate_hz;
    const uint32_t dt_us = (uint32_t)(1e6f / (float)q->cfg.gyro_rate_hz + 0.5f);
    for (int k = 0; k < n; k++) {
        q->now_us += dt_us;
        q->t += dt;
        vec3_t g, a;
        sense(q, &g, &a);
        if (fc_gyro_sample(&q->fc, g, a)) {
            fc_pid_step(&q->fc, q->now_us);
        }
        uint16_t d[4];
        float u[4];
        fc_motor_dshot(&q->fc, d);
        for (int i = 0; i < 4; i++) {
            u[i] = d[i] >= DSHOT_MIN_THROTTLE
                       ? (float)(d[i] - DSHOT_MIN_THROTTLE) / (float)DSHOT_RANGE
                       : 0.0f;
        }
        physics(q, u, dt);
        if ((int32_t)(q->now_us - s->next_telem_us) >= 0) {
            s->next_telem_us += 100000u;
            telemetry(s);
        }
    }
}

/* out: t, pos[3], vel[3], q[4] (w,x,y,z), w[3] rad/s, armed, fs, motor[4],
 *      rate_sp[3] deg/s, on_ground, throttle, rc_frames, arm_block      (27) */
#define SITL_STATE_LEN 27
SITL_API int sitl_state_len(void)
{
    return SITL_STATE_LEN;
}

SITL_API void sitl_get(const sitl_t *s, float *out)
{
    const quad_sim_t *q = &s->sim;
    int i = 0;
    out[i++] = q->t;
    out[i++] = q->pos.x; out[i++] = q->pos.y; out[i++] = q->pos.z;
    out[i++] = q->vel.x; out[i++] = q->vel.y; out[i++] = q->vel.z;
    out[i++] = q->q.w; out[i++] = q->q.x; out[i++] = q->q.y; out[i++] = q->q.z;
    out[i++] = q->w.x; out[i++] = q->w.y; out[i++] = q->w.z;
    out[i++] = q->fc.armed ? 1.0f : 0.0f;
    out[i++] = (float)q->fc.fs;
    for (int m = 0; m < 4; m++) out[i++] = q->fc.motor[m];
    out[i++] = q->fc.rate_sp.x; out[i++] = q->fc.rate_sp.y; out[i++] = q->fc.rate_sp.z;
    out[i++] = q->on_ground ? 1.0f : 0.0f;
    out[i++] = q->fc.throttle;
    out[i++] = (float)s->rc_frames;
    out[i++] = (float)q->fc.arm_block;
}

/* ---- thin exports of individual skyforge-fc functions (conformance tests) ---- */
SITL_API uint8_t sitl_crc8(const uint8_t *d, int n) { return crsf_crc8(d, n); }
SITL_API void sitl_pack(const uint16_t *ch, uint8_t *out) { crsf_pack_channels(ch, out); }
SITL_API void sitl_unpack(const uint8_t *pl, uint16_t *ch) { crsf_unpack_channels(pl, ch); }
SITL_API float sitl_rate_curve(float stick, float c, float m, float e) { return rc_rate_curve(stick, c, m, e); }
SITL_API float sitl_ch_to_norm(uint16_t v) { return crsf_ch_to_norm(v); }
SITL_API int sitl_mix(float thr, float r, float p, float y, int airmode, float *out)
{
    mixer_t m;
    mixer_init_quadx(&m, airmode != 0, false);
    return mixer_mix(&m, thr, r, p, y, out) ? 1 : 0;
}
SITL_API int sitl_dshot_value(float motor, float idle) { return dshot_value_from_motor(motor, idle); }
SITL_API float sitl_pid_run(const float *sp, const float *meas, int n, float kp, float ki, float kd,
                            float kf, float hz, float *out)
{
    pid_axis_t a;
    pid_gains_t g = {kp, ki, kd, kf};
    pid_axis_init(&a, g, hz, 100.0f, 40.0f);
    for (int i = 0; i < n; i++) {
        out[i] = pid_axis_update(&a, sp[i], meas[i], 1.0f / hz, 1.0f, false);
    }
    return a.integ;
}
