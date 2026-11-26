# Algorithms and derivations

Conventions everywhere: world frame **NED**, body frame **FRD** (as skyforge-fc),
quaternions `(w, x, y, z)` body→world, gravity `+g e3`. Camera frame is OpenCV
(x right, y down, z forward), mounted with a 20° up-tilt.

## 1. Gate detection (GateNet)

**Task.** For each image, find the four *inner* corners of every gate, typed
TL, TR, BR, BL in the gate's own frame (up = away from its legs, left/right as
seen from the approach side). A gate seen from behind is mirrored, so left and
right swap; the renderer and the association step use the same rule
(`Gate.inner_corners_local(mirrored)`). In a banked turn (the closed loop sees
up to ~85° of roll) the types rotate with the image, so the network has to use
the legs and the horizon; the classical baseline cannot type corners there and
is associated type-agnostically instead.

**Network.** 7 conv3×3 + ReLU layers (stride 2 three times, one dilated layer for
context, a nearest-upsample + skip connection back to stride 4) and a 1×1 head:
74.5 k parameters, 4 output heatmaps at 36×48 for a 144×192 input
([`skyrace/vision/cnn.py`](../skyrace/vision/cnn.py)).

**Targets and loss.** Each corner is a Gaussian (σ = 1 cell) on its type's
heatmap. Loss = CenterNet's penalty-reduced focal loss [Zhou 2019]:

```
L = -1/N_pos [ Σ_pos (1-p)^2 log p  +  Σ_neg (1-y)^4 p^2 log(1-p) ]
dL/dz (pos) = 2 p (1-p)^2 log p - (1-p)^3
dL/dz (neg) = -(1-y)^4 [ 2 p^2 (1-p) log(1-p) - p^3 ]
```

Corners further than 22 m are "don't care" (zero weight around them).

**Training is hand-written backprop in numpy.** Every layer is one im2col
(`as_strided`, layout `(C, B, H, W)` so no transposes) + one GEMM forward; the
backward pass is the transposed GEMMs plus `col2im` as 9 strided accumulations,
upsampling-backward as a 2×2 block sum, and the skip connection's gradient
added where the branch splits. `tests/test_vision.py` checks every parameter
block against central finite differences. Adam, warm-up + cosine LR,
horizontal-flip augmentation that swaps TL↔TR / BL↔BR.

**Decoding.** 3×3 local maxima above 0.3, refined by the 3×3 centroid.

**Classical baseline** ([`classical.py`](../skyrace/vision/classical.py)):
orange chromaticity mask → connected components → the enclosed hole of each
component (connected components of the inverted mask not touching the bounding
box) → corners as extreme points along ±(u+v), ±(u−v).

## 2. Planar PnP ([`pnp.py`](../skyrace/vision/pnp.py))

Gate corners lie on the gate plane `x_g = 0`. With plane coordinates `(y, z)`:

```
s [u_n, v_n, 1]^T = [r2 r3 t] [y, z, 1]^T          (normalised image coords)
```

1. Hartley-normalised DLT → `H`; `λ = 2 / (|h1| + |h2|)`, sign so that `t_z > 0`.
2. `r2 = λ h1, r3 = λ h2, r1 = r2 × r3`, project `[r1 r2 r3]` to SO(3) by SVD.
3. Levenberg–Marquardt on the reprojection error, `R ← exp([δθ]×) R`:
   `∂(R X + t)/∂δθ = -[R X]×`, `∂/∂t = I`, pinhole Jacobian
   `[[f/z, 0, -f x/z²], [0, f/z, -f y/z²]]`.
4. `Σ_(δθ,δt) = σ_px² (JᵀJ)⁻¹`.

**From camera pose to the EKF measurement.** With `R_cg, t` (gate→camera):
`R_wb = R_wg R_cgᵀ R_bcᵀ`, `p = c_g − R_wg R_cgᵀ t`. Perturbing `R_cg` on the
left by `δθ` and `t` by `δt` gives, to first order,

```
δp     = -R_wc ([t]× δθ + δt)
δφ_b   = -R_bc δθ                (body-frame right perturbation, R_wb exp([δφ_b]×))
```

so the 6×6 measurement covariance is `J Σ Jᵀ` with
`J = [[-R_wc [t]×, -R_wc], [-R_bc, 0]]`. The rotation–translation coupling of a
small planar target shows up as strong off-diagonal terms; the EKF uses the
full matrix, and its tight IMU attitude prior then pins the position well.
A Monte-Carlo test checks the covariance at 3 m (`test_pnp_noise_and_covariance_consistent`).

## 3. Error-state EKF ([`ekf.py`](../skyrace/estimation/ekf.py))

Nominal state `(p, v, R, b_g, b_a)`; 15-D error state `(δp, δv, δθ, δb_g, δb_a)`
with `R = R̂ exp([δθ]×)`. IMU model `ω_m = ω + b_g + n_g`,
`f_m = Rᵀ(a − g e3) + b_a + n_a`.

```
δṗ = δv
δv̇ = -R [f]× δθ - R δb_a - R n_a
δθ̇ = -[ω]× δθ - δb_g - n_g
δḃ = n_b
```

Discretised as `F = I + A dt`, `Q_d = G Q_c Gᵀ dt`. Pose update with residual
`r = [p_m − p̂ ; log(R̂ᵀ R_m)]`, `H = [[I 0 0 0 0], [0 0 I 0 0]]`, Joseph-form
covariance update, χ²(6) gate at 22.46 (99.9 %). Zero-velocity updates while
on the pad. Measurement association: every gate whose four predicted corners
fall in the image (from the EKF prior) is tried; per corner type the nearest
detection within `8 px + 3 σ_p f / depth` is used.

This is lighter than insight-vio's MSCKF on purpose: the gates are a known map,
so one PnP is an absolute 6-DoF fix and no feature states are needed. Between
gates the filter dead-reckons on the IMU (~1 s at most on this track).

## 4. Minimum-snap with time allocation ([`minsnap.py`](../skyrace/planning/minsnap.py))

Per axis, segment `i` is `p_i(τ) = Σ_{n=0}^{7} c_n τⁿ`, `τ ∈ [0, T_i]`.

* Snap cost: `J_i = c_iᵀ Q_i c_i`, `Q_i[j,k] = j!/(j-4)! · k!/(k-4)! · T^{j+k-7}/(j+k-7)` for `j,k ≥ 4`.
* Endpoint derivatives (orders 0–3 at both ends) `d_i = A_i c_i`.
* A selection matrix `S` maps the vector `D` of unique knot derivatives to the
  stacked `d_i`, which enforces C³ continuity by construction:
  `J = Dᵀ Sᵀ blkdiag(A_i⁻ᵀ Q_i A_i⁻¹) S D = Dᵀ R D`.
* Fixed `D_F`: all knot positions, start at rest. Free `D_P`: interior
  velocity/acceleration/jerk and the final derivatives. Setting `∂J/∂D_P = 0`:

```
D_P* = -R_PP⁻¹ R_PF D_F                        [Richter, Bry & Roy 2016]
```

**Time allocation.** All non-position constraints are zero, so scaling every
`T_i` by `α` keeps the path and scales `v, a, j` by `α⁻¹, α⁻², α⁻³`. For a
ratio vector `r`, the lap time is `α*(r) Σ r` where `α*` is found by bisection
as the smallest scale satisfying, along the trajectory,

```
T_min ≤ ‖m (a − g e3) + c v‖ ≤ T_max         (collective thrust incl. linear drag)
‖(I − z zᵀ)(m j + c a)‖ / T ≤ ω_max          (body-rate magnitude from flatness)
‖v‖ ≤ v_max
```

and `log r` is optimised by central-difference gradient descent with
backtracking (the gradient's mean is removed: uniform scaling is a null
direction). The planner's `T_max` is the knob for aggressiveness.

## 5. Differential flatness ([`reference.py`](../skyrace/planning/reference.py))

`f = m (a − g e3) + c v = −T R e3` ⇒ `T = ‖f‖`, `z_b = −f/T`,
`y_b = z_b × x_c / ‖·‖` with `x_c = (cos ψ, sin ψ, 0)`, `x_b = y_b × z_b`.
Yaw ψ follows the velocity heading (the FPV camera looks along the flight
path). Body rates come from the sampled rotations, `ω_k = log(R_kᵀ R_{k+1}) / dt`.

## 6. Controllers ([`controllers.py`](../skyrace/control/controllers.py))

All three output collective thrust + body rates, which is what skyforge-fc's
acro mode consumes.

**Flatness tracker.** `a_cmd = a_ref + K_p (p_ref − p̂) + K_v (v_ref − v̂)`;
desired attitude from `a_cmd` and ψ_ref; `T = f · (−R̂ e3)`;
`ω = R̂ᵀ R_d ω_ref + k_R log(R̂ᵀ R_d)`.

**LTV-MPC.** Error state `e = (δp, δv, δθ)` with `R = R_ref exp([δθ]×)`, input
`u = (δT, δω)`, linearised about the reference at every horizon step:

```
δṗ = δv
δv̇ = -(c/m) δv + (T_ref/m) R_ref [e3]× δθ − (1/m) R_ref e3 δT
δθ̇ = -[ω_ref]× δθ + δω
```

discretised with a 2nd-order expansion (`dt = 40 ms`, `N = 16`), condensed
(`X = Φ e0 + Γ U`) into `min ½ Uᵀ H U + gᵀ U` with box constraints
`T_min ≤ T_ref + δT ≤ T_max`, `|ω_ref + δω| ≤ ω_max`, solved at 50 Hz by the
ADMM QP below, warm-started with the shifted previous solution. A 30 ms
constant-acceleration prediction compensates the command latency.

**ADMM QP** ([`qp.py`](../skyrace/control/qp.py)) – the OSQP iteration
[Stellato 2020] on `min ½xᵀPx + qᵀx s.t. l ≤ Ax ≤ u` with relaxation 1.6,
residual-based stopping and ρ adaptation (re-factorising the small dense KKT
matrix).

**Pure pursuit (baseline).** A polyline through every gate along its normal
(±2 m approach points); chase the point `L = 3 m + 0.3 s · |v|` ahead at a fixed
cruise speed with a velocity P-loop, same thrust/attitude mapping.

## 7. Commands → CRSF ([`commands.py`](../skyrace/link/commands.py))

skyforge-fc acro: `rate = rc_rate_curve(deadband(stick))` (pitch negated),
`motor = mixer(throttle, PID(rate − gyro))`. The companion inverts each step:

* `rc_rate_curve` is odd and strictly increasing
  (`s·c + (max−c)·s|s|((1−e) + e s⁴)`), inverted by safeguarded Newton;
* the deadband rescaling is inverted exactly;
* thrust → throttle from `T = 4 T_max (idle + (1−idle) thr)²`;
* stick → 11-bit channel `992 + 819 s`, packed LSB-first into the 22-byte
  RC_CHANNELS payload with CRC-8/DVB-S2 (poly 0xD5).

Quantisation is < 1.5 °/s at full stick and 0.06 N in thrust (tested).

## References

* D. Mellinger, V. Kumar, "Minimum snap trajectory generation and control for quadrotors", ICRA 2011.
* C. Richter, A. Bry, N. Roy, "Polynomial trajectory planning for aggressive quadrotor flight in dense indoor environments", ISRR 2013 / Springer 2016.
* M. Faessler, A. Franchi, D. Scaramuzza, "Differential flatness of quadrotor dynamics subject to rotor drag", RA-L 2018.
* B. Stellato et al., "OSQP: an operator splitting solver for quadratic programs", Math. Prog. Comp. 2020.
* X. Zhou, D. Wang, P. Krähenbühl, "Objects as Points", arXiv:1904.07850, 2019.
* Z. Zhang, "A flexible new technique for camera calibration", PAMI 2000.
* E. Kaufmann et al., "Champion-level drone racing using deep reinforcement learning", Nature 2023 (context: gate-corner detection + filtering is the classical pipeline it competes with).
* P. Foehn et al., "AlphaPilot: autonomous drone racing", RSS 2020 (corner heatmaps + association).
* J. Solà, "Quaternion kinematics for the error-state Kalman filter", arXiv:1711.02508, 2017.
