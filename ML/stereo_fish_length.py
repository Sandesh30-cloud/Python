# #!/usr/bin/env python3
# """
# stereo_fish_length.py - robust version

# Improvements over original:
#  - SGBM tuned and clamped more robustly
#  - disparity postprocessing (median blur + speckle removal)
#  - endpoint selection improved using convex hull longest-pair
#  - line sampling collects many samples and uses trimmed median/mean
#  - avoids trusting single extreme endpoint disparities
#  - clearer diagnostics and CLI tuning parameters

# Usage:
#     python3 stereo_fish_length.py --left left.png --right right.png --baseline-cm 8.0
# """
# import argparse
# import os
# from pathlib import Path
# import numpy as np
# import cv2
# import sys

# # ----------------- Sensor / Defaults / Tunables -----------------
# SENSOR_NATIVE_W = 2592
# SENSOR_NATIVE_H = 1944
# PIXEL_SIZE_MM = 0.0014   # 1.4 µm = 0.0014 mm
# DEFAULT_LENS_F_MM = 3.6
# DEFAULT_CALIB = "stereo_calib.npz"
# BASELINE_CM_DEFAULT = 22.0
# NUM_DISP_MULT = 12             # default multiplier (was 28)
# MIN_ACCEPTABLE_DISP_PX = 4.0   # default minimum acceptable disparity (px)
# TRIM_PERCENT = 20              # trim this percent of extreme values from both ends when computing robust stat
# MAX_REASONABLE_DISP = 1000.0   # clip/spike detection (px)
# # -------------------------------------------------------

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--left", required=True)
#     p.add_argument("--right", required=True)
#     p.add_argument("--calib", default=DEFAULT_CALIB)
#     p.add_argument("--baseline-cm", type=float, default=BASELINE_CM_DEFAULT)
#     p.add_argument("--no-yolo", action="store_true")
#     p.add_argument("--out-vis", default="left_vis_measured.png")
#     p.add_argument("--out-disp", default="disp_vis_measured.png")
#     p.add_argument("--verbose", action="store_true")
#     p.add_argument("--num-disp-mult", type=int, default=NUM_DISP_MULT,
#                    help="multiplier for numDisparities: numDisparities = 16 * num_disp_mult")
#     p.add_argument("--min-acceptable-disp-px", type=float, default=MIN_ACCEPTABLE_DISP_PX)
#     p.add_argument("--trim-percent", type=float, default=TRIM_PERCENT,
#                    help="percent trimmed from both ends when computing robust statistic on samples")
#     return p.parse_args()

# def load_images(left_path, right_path):
#     left = cv2.imread(str(left_path))
#     right = cv2.imread(str(right_path))
#     if left is None or right is None:
#         raise FileNotFoundError(f"Left or right image not found: {left_path}, {right_path}")
#     return left, right

# def compute_fx_guess_for_size(img_w, img_h, lens_f_mm=DEFAULT_LENS_F_MM):
#     sensor_w_mm = SENSOR_NATIVE_W * PIXEL_SIZE_MM
#     fx_px = lens_f_mm * (img_w / sensor_w_mm)
#     return float(fx_px)

# def save_calib_synthetic(filename, img_size, fx=None, baseline_m=BASELINE_CM_DEFAULT/100.0):
#     w, h = img_size
#     if fx is None:
#         fx = compute_fx_guess_for_size(w, h, lens_f_mm=DEFAULT_LENS_F_MM)
#     K = np.array([[fx, 0, w/2.0],
#                   [0, fx, h/2.0],
#                   [0,  0,    1.0]], dtype=np.float64)
#     D = np.zeros(5, dtype=np.float64)
#     R = np.eye(3, dtype=np.float64)
#     T = np.array([baseline_m, 0.0, 0.0], dtype=np.float64)
#     R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(K, D, K, D, (w, h), R, T, alpha=0)
#     map1x,map1y = cv2.initUndistortRectifyMap(K,D,R1,P1,(w,h),cv2.CV_32FC1)
#     map2x,map2y = cv2.initUndistortRectifyMap(K,D,R2,P2,(w,h),cv2.CV_32FC1)
#     np.savez(filename,
#              K1=K, D1=D, K2=K, D2=D, R=R, T=T,
#              R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
#              map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
#              img_size=(w,h))
#     print(f"[INFO] Saved synthetic calibration to {filename} (fx={fx:.1f}px, baseline={baseline_m:.3f} m)")

# def load_calib(filename):
#     if not os.path.exists(filename):
#         return None
#     npz = np.load(filename, allow_pickle=True)
#     data = {k: npz[k] for k in npz.files}
#     print(f"[INFO] Loaded calibration from {filename}")
#     return data

# def rectify_pair(left, right, calib):
#     if calib is None:
#         raise ValueError("Calibration required for rectification")
#     map1x, map1y = calib["map1x"], calib["map1y"]
#     map2x, map2y = calib["map2x"], calib["map2y"]
#     left_rect = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_LINEAR)
#     right_rect = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_LINEAR)
#     return left_rect, right_rect

# def _clamp_num_disp_to_width(num_disp, img_w):
#     if num_disp <= 0:
#         num_disp = 16
#     if num_disp % 16 != 0:
#         num_disp = (num_disp // 16 + 1) * 16
#     max_allowed = max(16, (img_w // 16) * 16)
#     if num_disp > max_allowed:
#         num_disp = max_allowed
#     return int(num_disp)

# def compute_disparity(left_gray, right_gray, num_disp=None, verbose=False):
#     window_size = 7
#     min_disp = 0
#     h, w = left_gray.shape[:2]
#     if num_disp is None:
#         num_disp = 16 * NUM_DISP_MULT
#     num_disp = _clamp_num_disp_to_width(num_disp, w)
#     # SGBM tuned for robustness: lower uniquenessRatio, larger speckleWindow, moderate P1/P2
#     stereo = cv2.StereoSGBM_create(
#         minDisparity = min_disp,
#         numDisparities = num_disp,
#         blockSize = window_size,
#         P1 = 8*3*window_size**2,
#         P2 = 32*3*window_size**2,
#         disp12MaxDiff = 2,
#         uniquenessRatio = 5,
#         speckleWindowSize = 200,
#         speckleRange = 16,
#         preFilterCap = 63,
#         mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
#     )
#     disp_raw = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
#     # Post-process: mask invalid, median blur to remove spikes, remove tiny speckles
#     disp = disp_raw.copy()
#     # replace negative and large values with NaN
#     disp[(disp <= 0) | (~np.isfinite(disp)) | (disp > MAX_REASONABLE_DISP)] = np.nan
#     # median blur ignoring NaNs - approximate by filling NaNs with local median using a sliding window
#     disp_filled = disp.copy()
#     nan_mask = np.isnan(disp_filled)
#     if np.any(nan_mask):
#         # simple inpainting: replace NaN with local median using a moving window
#         k = 7
#         padded = np.pad(disp_filled, ((k,k),(k,k)), mode='constant', constant_values=np.nan)
#         out = np.full_like(disp_filled, np.nan)
#         for y in range(disp_filled.shape[0]):
#             for x in range(disp_filled.shape[1]):
#                 patch = padded[y:y+2*k+1, x:x+2*k+1]
#                 vals = patch[np.isfinite(patch)]
#                 if vals.size > 0:
#                     out[y,x] = np.median(vals)
#         disp_filled = out
#     # final median blur to smooth residual noise
#     disp_smooth = cv2.medianBlur(np.nan_to_num(disp_filled, nan=0.0).astype(np.float32), 5).astype(np.float32)
#     # restore NaNs where we originally had nonefinite after smoothing threshold
#     # anything that remains zero and original had no finite estimate, keep NaN
#     mask_zero_but_orig_nan = (disp_smooth == 0.0) & np.isnan(disp_filled)
#     disp_smooth = disp_smooth.astype(np.float32)
#     disp_smooth[mask_zero_but_orig_nan] = np.nan
#     # remove small speckles by morphological operations on finite mask
#     finite_mask = np.isfinite(disp_smooth).astype(np.uint8)*255
#     kernel = np.ones((5,5), np.uint8)
#     finite_mask = cv2.morphologyEx(finite_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     disp_smooth[finite_mask==0] = np.nan
#     if verbose:
#         print(f"[DIAG] disparity min/max (pre) = {np.nanmin(disp_raw):.2f}/{np.nanmax(disp_raw):.2f}")
#         finite = np.isfinite(disp_smooth)
#         if np.any(finite):
#             print(f"[DIAG] disparity min/max (post) = {np.nanmin(disp_smooth[finite]):.2f}/{np.nanmax(disp_smooth[finite]):.2f}")
#         else:
#             print("[DIAG] disparity map has no finite values after postproc")
#     return disp_smooth

# def try_yolo_bbox(left_img):
#     try:
#         from ultralytics import YOLO
#     except Exception as e:
#         return None, f"ultralytics import failed: {e}"
#     try:
#         model = YOLO("yolov8n.pt")
#         # predict returns results list; pass image array or path as source
#         results = model.predict(source=left_img, imgsz=640, conf=0.25, verbose=False)
#     except Exception as e:
#         return None, f"YOLO predict failed: {e}"
#     best_box = None
#     best_score = -1.0
#     for r in results:
#         if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
#             continue
#         xyxy = r.boxes.xyxy.cpu().numpy()
#         confs = r.boxes.conf.cpu().numpy()
#         for (x1,y1,x2,y2), conf in zip(xyxy, confs):
#             if conf > best_score:
#                 best_score = float(conf)
#                 best_box = (int(x1), int(y1), int(x2), int(y2))
#     if best_box is None:
#         return None, "no boxes found"
#     return best_box, None

# def detect_bbox_from_disp(disp, percentile=92):
#     mask = np.isfinite(disp) & (disp > 0)
#     if not np.any(mask):
#         return None
#     vals = disp[mask]
#     thresh = np.percentile(vals, percentile)
#     obj_mask = (disp >= thresh).astype(np.uint8)
#     obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
#     contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None
#     cnt = max(contours, key=cv2.contourArea)
#     x,y,w,h = cv2.boundingRect(cnt)
#     pad = 5
#     return (max(0,x-pad), max(0,y-pad), x+w+pad, y+h+pad), cnt

# def detect_bbox_contour(gray):
#     blur = cv2.GaussianBlur(gray, (7,7), 0)
#     _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     kernel = np.ones((5,5), np.uint8)
#     th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
#     contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None
#     cnt = max(contours, key=cv2.contourArea)
#     x,y,w,h = cv2.boundingRect(cnt)
#     pad = 5
#     return (max(0,x-pad), max(0,y-pad), x+w+pad, y+h+pad), cnt

# def endpoints_from_contour_using_hull(cnt):
#     """Return the two most-distant points on the contour's convex hull (robust to small contour noise)."""
#     if cnt is None or len(cnt) == 0:
#         return None
#     hull = cv2.convexHull(cnt, returnPoints=True)
#     pts = hull.reshape(-1,2)
#     n = len(pts)
#     if n < 2:
#         return None
#     # brute-force O(n^2) pair search on hull points (hull is typically small)
#     maxd = 0.0
#     p1 = p2 = None
#     for i in range(n):
#         for j in range(i+1, n):
#             d = np.sum((pts[i] - pts[j])**2)
#             if d > maxd:
#                 maxd = d
#                 p1 = tuple(pts[i])
#                 p2 = tuple(pts[j])
#     return (p1, p2)

# def endpoints_from_bbox(gray, bbox):
#     x1,y1,x2,y2 = [int(v) for v in bbox]
#     crop = gray[y1:y2, x1:x2]
#     if crop.size == 0:
#         return None
#     blur = cv2.GaussianBlur(crop, (5,5), 0)
#     _, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     kernel = np.ones((3,3), np.uint8)
#     th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
#     contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None
#     cnt = max(contours, key=cv2.contourArea)
#     hull = cv2.convexHull(cnt)
#     res = endpoints_from_contour_using_hull(hull)
#     if res is None:
#         return None
#     (p1_rel, p2_rel) = res
#     p1 = (int(p1_rel[0] + x1), int(p1_rel[1] + y1))
#     p2 = (int(p2_rel[0] + x1), int(p2_rel[1] + y1))
#     return (p1, p2), cnt

# def robust_disp_at(disp, u, v, max_win=21):
#     h,w = disp.shape
#     u_c = np.clip(int(round(u)), 0, w-1)
#     v_c = np.clip(int(round(v)), 0, h-1)
#     for win in range(3, max_win+1, 2):
#         x0 = max(0, u_c - win); x1 = min(w, u_c + win + 1)
#         y0 = max(0, v_c - win); y1 = min(h, v_c + win + 1)
#         patch = disp[y0:y1, x0:x1]
#         patch = patch[np.isfinite(patch) & (patch > 0)]
#         if patch.size > 0:
#             return float(np.median(patch))
#     return None

# def line_disp_samples(disp, p1, p2, n=101, win=11):
#     samples = []
#     (u1,v1) = p1; (u2,v2) = p2
#     for i in range(n):
#         t = i / max(1, n-1)
#         u = u1 * (1 - t) + u2 * t
#         v = v1 * (1 - t) + v2 * t
#         d = robust_disp_at(disp, u, v, max_win=win)
#         if d is not None and d > 0:
#             samples.append(d)
#     return samples

# def compute_trimmed_stat(samples, trim_percent=TRIM_PERCENT):
#     if len(samples) == 0:
#         return None
#     arr = np.sort(np.asarray(samples, dtype=np.float64))
#     k = int(len(arr) * (trim_percent/100.0))
#     if k*2 >= len(arr):
#         return float(np.median(arr))
#     trimmed = arr[k:len(arr)-k]
#     if trimmed.size == 0:
#         return float(np.median(arr))
#     return float(np.median(trimmed))

# def pixel_disp_to_xyz(u, v, d_px, calib=None, fx=None, cx=None, cy=None, baseline_m=None, baseline_calib=None):
#     if d_px is None or d_px <= 0 or not np.isfinite(d_px):
#         return None
#     use_override_baseline = False
#     if baseline_m is not None and baseline_calib is not None:
#         diff_ratio = abs(baseline_m - baseline_calib) / max(baseline_calib, 1e-6)
#         if diff_ratio > 0.05:
#             use_override_baseline = True
#     if calib is not None and "Q" in calib and not use_override_baseline:
#         Q = calib["Q"]
#         vec = np.array([u, v, d_px, 1.0], dtype=np.float64)
#         X, Y, Z, W = (Q @ vec)[:4]
#         if abs(W) < 1e-9:
#             return None
#         return np.array([X/W, Y/W, Z/W])
#     if fx is None or cx is None or cy is None or baseline_m is None:
#         return None
#     Z = (fx * baseline_m) / d_px
#     X = (u - cx) * Z / fx
#     Y = (v - cy) * Z / fx
#     return np.array([X, Y, Z])

# def diag_print_calib(calib, baseline_m_override, w, h):
#     fx_guess = compute_fx_guess_for_size(w, h, lens_f_mm=DEFAULT_LENS_F_MM)
#     cx_guess, cy_guess = w / 2.0, h / 2.0
#     if calib is None:
#         fx = fx_guess; cx, cy = cx_guess, cy_guess; baseline = None
#         print(f"[DIAG] No calib. Using fx≈{fx:.1f}, cx={cx:.1f}, cy={cy:.1f}, baseline_override={baseline_m_override:.3f} m")
#         for Zm in (0.4,0.8,1.5):
#             d_ex = fx * baseline_m_override / Zm
#             print(f"[DIAG] expected disparity at Z={Zm:.2f} m: d ≈ {d_ex:.1f} px")
#         return fx, cx, cy, None
#     K = None
#     for key in ("K1","K","cameraMatrix1","camera_matrix_left"):
#         if key in calib:
#             K = calib[key]; break
#     if K is None:
#         fx = fx_guess; cx, cy = cx_guess, cy_guess
#     else:
#         try:
#             fx = float(K[0,0]); cx = float(K[0,2]); cy = float(K[1,2])
#         except Exception:
#             fx = fx_guess; cx, cy = cx_guess, cy_guess
#     baseline_calib = None
#     if "T" in calib:
#         try:
#             T = calib["T"]
#             baseline_calib = float(np.linalg.norm(np.asarray(T).reshape(-1)[:3]))
#         except Exception:
#             baseline_calib = None
#     print(f"[DIAG] fx={fx:.1f}px, cx={cx:.1f}, cy={cy:.1f}, baseline_in_calib={baseline_calib}, baseline_override={baseline_m_override:.3f} m")
#     baseline_for_calc = baseline_calib if baseline_calib is not None else baseline_m_override
#     for Zm in (0.4,0.8,1.5):
#         d_ex = fx * baseline_for_calc / Zm
#         print(f"[DIAG] expected disparity at Z={Zm:.2f} m: d ≈ {d_ex:.1f} px")
#     return fx, cx, cy, baseline_calib

# def main():
#     args = parse_args()
#     left_path = Path(args.left)
#     right_path = Path(args.right)
#     calib_file = args.calib
#     baseline_m_override = args.baseline_cm / 100.0

#     left, right = load_images(left_path, right_path)
#     h, w = left.shape[:2]
#     fx_guess_local = compute_fx_guess_for_size(w, h, lens_f_mm=DEFAULT_LENS_F_MM)

#     calib = load_calib(calib_file)
#     if calib is None:
#         print("[WARN] Calibration file not found. Creating synthetic calibration for quick testing (not accurate).")
#         save_calib_synthetic(calib_file, (w, h), fx=fx_guess_local, baseline_m=baseline_m_override)
#         calib = load_calib(calib_file)

#     fx, cx, cy, baseline_calib = diag_print_calib(calib, baseline_m_override, w, h)

#     left_rect, right_rect = rectify_pair(left, right, calib)
#     left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
#     right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

#     requested_num_disp = 16 * args.num_disp_mult
#     requested_num_disp = _clamp_num_disp_to_width(requested_num_disp, w)
#     disp = compute_disparity(left_gray, right_gray, num_disp=requested_num_disp, verbose=args.verbose)

#     # Save disparity visualization (scale finite values)
#     vis_disp = np.zeros_like(left_gray, dtype=np.float32)
#     finite = np.isfinite(disp)
#     if np.any(finite):
#         dmin = float(np.nanmin(disp[finite])); dmax = float(np.nanmax(disp[finite]))
#         vis_disp[finite] = (disp[finite] - dmin) / (max(1e-9, dmax - dmin))
#     cv2.imwrite(args.out_disp, (vis_disp * 255).astype(np.uint8))
#     print(f"[INFO] Saved disparity visualization to {args.out_disp}")

#     bbox = None
#     contour_for_vis = None
#     if not args.no_yolo:
#         bbox, yolo_err = try_yolo_bbox(left_rect)
#         if bbox is None:
#             print(f"[INFO] YOLO not used/found: {yolo_err}")
#         else:
#             print(f"[INFO] YOLO bbox detected: {bbox}")

#     if bbox is None:
#         res = detect_bbox_from_disp(disp, percentile=92)
#         if res is not None:
#             bbox, contour_for_vis = res
#             print(f"[INFO] Disparity-based bbox used: {bbox}")

#     if bbox is None:
#         res = detect_bbox_contour(left_gray)
#         if res is None:
#             print("[ERROR] No detection/bbox found. Exiting.")
#             return
#         bbox, contour_for_vis = res
#         print(f"[INFO] Contour-based bbox used: {bbox}")

#     endpoints_cnt = endpoints_from_bbox(left_gray, bbox)
#     if endpoints_cnt is None:
#         print("[ERROR] Could not find contour endpoints inside bbox.")
#         return
#     (p1, p2), cnt = endpoints_cnt
#     if p1 is None or p2 is None:
#         print("[ERROR] endpoints computation failed.")
#         return
#     u1, v1 = int(p1[0]), int(p1[1])
#     u2, v2 = int(p2[0]), int(p2[1])
#     print(f"[INFO] Endpoint pixels (hull longest pair): p1=({u1},{v1}), p2=({u2},{v2})")

#     # Sample disparities along the line (dense sampling) and compute trimmed median
#     samples = line_disp_samples(disp, (u1, v1), (u2, v2), n=201, win=11)
#     robust_val = compute_trimmed_stat(samples, trim_percent=args.trim_percent)
#     # Also compute robust stats of endpoint neighborhoods separately
#     d1 = robust_disp_at(disp, u1, v1, max_win=41)
#     d2 = robust_disp_at(disp, u2, v2, max_win=41)
#     print(f"[INFO] Disparities (px): d1={d1}, d2={d2}, samples_count={len(samples)}, trimmed_stat={robust_val}")

#     # Decide final disparity to use: prefer trimmed stat if it is reasonable, else fall back to endpoints if consistent
#     MIN_ACCEPTABLE = args.min_acceptable_disp_px
#     chosen_disp = None
#     if robust_val is not None and robust_val > MIN_ACCEPTABLE and robust_val < MAX_REASONABLE_DISP:
#         chosen_disp = robust_val
#         reason = "trimmed_line_stat"
#     else:
#         # check endpoint consistency
#         if d1 is not None and d2 is not None and d1 > MIN_ACCEPTABLE and d2 > MIN_ACCEPTABLE:
#             ratio = abs(d1 - d2) / max(d1, d2, 1.0)
#             if ratio < 0.4 and max(d1,d2) < MAX_REASONABLE_DISP:
#                 chosen_disp = (d1 + d2) / 2.0
#                 reason = "endpoint_mean"
#         # fallback to whichever endpoint is valid
#         if chosen_disp is None:
#             if d1 is not None and d1 > MIN_ACCEPTABLE and d1 < MAX_REASONABLE_DISP:
#                 chosen_disp = d1; reason = "d1_fallback"
#             elif d2 is not None and d2 > MIN_ACCEPTABLE and d2 < MAX_REASONABLE_DISP:
#                 chosen_disp = d2; reason = "d2_fallback"

#     if chosen_disp is None:
#         print(f"[ERROR] disparity too small/unreliable for metric measurement (d1={d1}, d2={d2}, trimmed_stat={robust_val}).")
#         # Save debug visualization and return
#         vis = left_rect.copy()
#         x1,y1,x2,y2 = [int(v) for v in bbox]
#         cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
#         if contour_for_vis is not None:
#             cv2.drawContours(vis, [contour_for_vis], -1, (0,200,0), 2)
#         cv2.circle(vis, (u1,v1), 6, (0,0,255), -1)
#         cv2.circle(vis, (u2,v2), 6, (255,0,0), -1)
#         cv2.line(vis, (u1,v1), (u2,v2), (255,255,0), 2)
#         cv2.putText(vis, "DISP INVALID", (min(u1,u2), min(v1,v2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
#         cv2.imwrite(args.out_vis, vis)
#         print(f"[INFO] Saved visualization to {args.out_vis}")
#         return

#     print(f"[INFO] chosen disparity (px) = {chosen_disp:.3f} (reason: {reason})")

#     # Use calibration K if available, else fallback to guessed fx/cx/cy
#     K1 = calib.get("K1", None)
#     if K1 is not None:
#         fx = float(K1[0,0]); cx = float(K1[0,2]); cy = float(K1[1,2])
#     else:
#         fx = fx_guess_local; cx = w/2.0; cy = h/2.0
#     T = calib.get("T", None)
#     baseline_calib = float(np.linalg.norm(T)) if T is not None else None
#     baseline_m = baseline_m_override

#     # For length estimation, compute 3D coordinates of both endpoints using chosen_disp
#     p3d_1 = pixel_disp_to_xyz(u1, v1, chosen_disp, calib=calib, fx=fx, cx=cx, cy=cy, baseline_m=baseline_m, baseline_calib=baseline_calib)
#     p3d_2 = pixel_disp_to_xyz(u2, v2, chosen_disp, calib=calib, fx=fx, cx=cx, cy=cy, baseline_m=baseline_m, baseline_calib=baseline_calib)

#     if p3d_1 is None or p3d_2 is None:
#         print("[ERROR] Cannot compute 3D points from chosen disparity.")
#         return

#     length_m = np.linalg.norm(p3d_1 - p3d_2)
#     length_cm = length_m * 100.0
#     print(f"[RESULT] Estimated object length = {length_cm:.2f} cm")
#     print(f"[DIAG] P1 (m) = {p3d_1}, P2 (m) = {p3d_2}, baseline used = {baseline_m:.3f} m, fx = {fx:.1f}")

#     # visualization
#     vis = left_rect.copy()
#     x1,y1,x2,y2 = [int(v) for v in bbox]
#     cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
#     if contour_for_vis is not None:
#         cv2.drawContours(vis, [contour_for_vis], -1, (0,200,0), 2)
#     cv2.circle(vis, (u1,v1), 6, (0,0,255), -1)
#     cv2.circle(vis, (u2,v2), 6, (255,0,0), -1)
#     cv2.line(vis, (u1,v1), (u2,v2), (255,255,0), 2)
#     cv2.putText(vis, f"{length_cm:.1f} cm", (min(u1,u2), min(v1,v2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
#     cv2.imwrite(args.out_vis, vis)
#     print(f"[INFO] Saved visualization to {args.out_vis}")

# if __name__ == "__main__":
#     main()


# COMMMAND TO RUN:
# python3 stereo_fish_length.py --left image_test_left.png --right image_test_right.png --baseline-cm 22.0 2>&1 | tail -12










































# #!/usr/bin/env python3
# """
# stereo_fish_length_fixed.py

# Combines your working disparity code with corrected stereoscopic measurement.
# Uses the paper's formula for accurate distance/length calculation.

# Usage:
#     python3 stereo_fish_length_fixed.py --left image_test_left.png --right image_test_right.png \
#         --baseline-cm 10.0 --fov-deg 70.0
# """
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
import math

# ----------------- Configuration -----------------
SENSOR_NATIVE_W = 2592
SENSOR_NATIVE_H = 1944
PIXEL_SIZE_MM = 0.0014
DEFAULT_LENS_F_MM = 3.6
DEFAULT_CALIB = "stereo_calib.npz"
DEFAULT_BASELINE_CM = 10.0
DEFAULT_FOV_DEG = 70.0
NUM_DISP_MULT = 12
MIN_ACCEPTABLE_DISP_PX = 4.0
TRIM_PERCENT = 20
MAX_REASONABLE_DISP = 1000.0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--calib", default=DEFAULT_CALIB)
    p.add_argument("--baseline-cm", type=float, default=DEFAULT_BASELINE_CM,
                   help="Distance between camera centers in cm")
    p.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG,
                   help="Camera field of view in degrees")
    p.add_argument("--no-yolo", action="store_true")
    p.add_argument("--out-vis", default="fish_measurement_result.png")
    p.add_argument("--out-disp", default="disparity_map.png")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--num-disp-mult", type=int, default=NUM_DISP_MULT)
    p.add_argument("--min-acceptable-disp-px", type=float, default=MIN_ACCEPTABLE_DISP_PX)
    p.add_argument("--trim-percent", type=float, default=TRIM_PERCENT)
    return p.parse_args()

def load_images(left_path, right_path):
    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))
    if left is None or right is None:
        raise FileNotFoundError(f"Images not found: {left_path}, {right_path}")
    
    # Auto-resize if dimensions don't match
    if left.shape != right.shape:
        print(f"[WARN] Image sizes don't match: {left.shape} vs {right.shape}")
        h_left, w_left = left.shape[:2]
        h_right, w_right = right.shape[:2]
        target_h = min(h_left, h_right)
        target_w = min(w_left, w_right)
        left = cv2.resize(left, (target_w, target_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (target_w, target_h), interpolation=cv2.INTER_AREA)
        print(f"[INFO] Resized to: {target_w}x{target_h}")
    
    return left, right

def compute_fx_from_fov(img_w, fov_deg):
    """Calculate focal length in pixels from field of view"""
    fov_rad = math.radians(fov_deg)
    fx = img_w / (2 * math.tan(fov_rad / 2))
    return fx

def save_calib_from_fov(filename, img_size, baseline_m, fov_deg):
    """Create calibration using FOV instead of guessing focal length"""
    w, h = img_size
    fx = compute_fx_from_fov(w, fov_deg)
    
    K = np.array([[fx, 0, w/2.0],
                  [0, fx, h/2.0],
                  [0,  0,    1.0]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    T = np.array([baseline_m, 0.0, 0.0], dtype=np.float64)
    
    R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(K, D, K, D, (w, h), R, T, alpha=0)
    map1x,map1y = cv2.initUndistortRectifyMap(K,D,R1,P1,(w,h),cv2.CV_32FC1)
    map2x,map2y = cv2.initUndistortRectifyMap(K,D,R2,P2,(w,h),cv2.CV_32FC1)
    
    np.savez(filename,
             K1=K, D1=D, K2=K, D2=D, R=R, T=T,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
             img_size=(w,h))
    print(f"[INFO] Created calibration: fx={fx:.1f}px, baseline={baseline_m:.3f}m, FOV={fov_deg:.1f}°")

def load_calib(filename):
    if not os.path.exists(filename):
        return None
    npz = np.load(filename, allow_pickle=True)
    return {k: npz[k] for k in npz.files}

def rectify_pair(left, right, calib):
    if calib is None:
        raise ValueError("Calibration required")
    map1x, map1y = calib["map1x"], calib["map1y"]
    map2x, map2y = calib["map2x"], calib["map2y"]
    left_rect = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_LINEAR)
    return left_rect, right_rect

def _clamp_num_disp_to_width(num_disp, img_w):
    if num_disp <= 0:
        num_disp = 16
    if num_disp % 16 != 0:
        num_disp = (num_disp // 16 + 1) * 16
    max_allowed = max(16, (img_w // 16) * 16)
    return min(int(num_disp), max_allowed)

def compute_disparity(left_gray, right_gray, num_disp=None, verbose=False):
    window_size = 7
    min_disp = 0
    h, w = left_gray.shape[:2]
    if num_disp is None:
        num_disp = 16 * NUM_DISP_MULT
    num_disp = _clamp_num_disp_to_width(num_disp, w)
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8*3*window_size**2,
        P2=32*3*window_size**2,
        disp12MaxDiff=2,
        uniquenessRatio=5,
        speckleWindowSize=200,
        speckleRange=16,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    disp_raw = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp = disp_raw.copy()
    disp[(disp <= 0) | (~np.isfinite(disp)) | (disp > MAX_REASONABLE_DISP)] = np.nan
    
    # Median filtering
    disp_smooth = cv2.medianBlur(np.nan_to_num(disp, nan=0.0).astype(np.float32), 5)
    disp_smooth = disp_smooth.astype(np.float32)
    disp_smooth[disp_smooth == 0] = np.nan
    
    # Remove small speckles
    finite_mask = np.isfinite(disp_smooth).astype(np.uint8)*255
    kernel = np.ones((5,5), np.uint8)
    finite_mask = cv2.morphologyEx(finite_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    disp_smooth[finite_mask==0] = np.nan
    
    if verbose:
        finite = np.isfinite(disp_smooth)
        if np.any(finite):
            print(f"[DIAG] Disparity range: {np.nanmin(disp_smooth[finite]):.2f} to {np.nanmax(disp_smooth[finite]):.2f} px")
    
    return disp_smooth

def try_yolo_bbox(left_img):
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        results = model.predict(source=left_img, imgsz=640, conf=0.25, verbose=False)
    except Exception as e:
        return None, f"YOLO failed: {e}"
    
    best_box = None
    best_score = -1.0
    for r in results:
        if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (x1,y1,x2,y2), conf in zip(xyxy, confs):
            if conf > best_score:
                best_score = float(conf)
                best_box = (int(x1), int(y1), int(x2), int(y2))
    
    if best_box is None:
        return None, "no boxes found"
    return best_box, None

def detect_bbox_from_disp(disp, percentile=92):
    mask = np.isfinite(disp) & (disp > 0)
    if not np.any(mask):
        return None
    vals = disp[mask]
    thresh = np.percentile(vals, percentile)
    obj_mask = (disp >= thresh).astype(np.uint8)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    pad = 5
    return (max(0,x-pad), max(0,y-pad), x+w+pad, y+h+pad), cnt

def endpoints_from_contour_hull(cnt):
    if cnt is None or len(cnt) == 0:
        return None
    hull = cv2.convexHull(cnt, returnPoints=True)
    pts = hull.reshape(-1,2)
    if len(pts) < 2:
        return None
    
    maxd = 0.0
    p1 = p2 = None
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.sum((pts[i] - pts[j])**2)
            if d > maxd:
                maxd = d
                p1 = tuple(pts[i])
                p2 = tuple(pts[j])
    return (p1, p2)

def endpoints_from_bbox(gray, bbox):
    x1,y1,x2,y2 = [int(v) for v in bbox]
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    
    blur = cv2.GaussianBlur(crop, (5,5), 0)
    _, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    res = endpoints_from_contour_hull(hull)
    if res is None:
        return None
    
    (p1_rel, p2_rel) = res
    p1 = (int(p1_rel[0] + x1), int(p1_rel[1] + y1))
    p2 = (int(p2_rel[0] + x1), int(p2_rel[1] + y1))
    return (p1, p2), cnt

def robust_disp_at(disp, u, v, max_win=21):
    h,w = disp.shape
    u_c = np.clip(int(round(u)), 0, w-1)
    v_c = np.clip(int(round(v)), 0, h-1)
    for win in range(3, max_win+1, 2):
        x0 = max(0, u_c - win); x1 = min(w, u_c + win + 1)
        y0 = max(0, v_c - win); y1 = min(h, v_c + win + 1)
        patch = disp[y0:y1, x0:x1]
        patch = patch[np.isfinite(patch) & (patch > 0)]
        if patch.size > 0:
            return float(np.median(patch))
    return None

def line_disp_samples(disp, p1, p2, n=101, win=11):
    samples = []
    (u1,v1) = p1; (u2,v2) = p2
    for i in range(n):
        t = i / max(1, n-1)
        u = u1 * (1 - t) + u2 * t
        v = v1 * (1 - t) + v2 * t
        d = robust_disp_at(disp, u, v, max_win=win)
        if d is not None and d > 0:
            samples.append(d)
    return samples

def compute_trimmed_stat(samples, trim_percent=TRIM_PERCENT):
    if len(samples) == 0:
        return None
    arr = np.sort(np.asarray(samples, dtype=np.float64))
    k = int(len(arr) * (trim_percent/100.0))
    if k*2 >= len(arr):
        return float(np.median(arr))
    trimmed = arr[k:len(arr)-k]
    if trimmed.size == 0:
        return float(np.median(arr))
    return float(np.median(trimmed))

def calculate_distance_from_disparity(disparity_px, baseline_cm, fx_px):
    """
    Calculate distance using: Z = (fx * baseline) / disparity
    Based on standard stereo vision formula
    """
    if disparity_px <= 0:
        return None
    baseline_m = baseline_cm / 100.0
    distance_m = (fx_px * baseline_m) / disparity_px
    return distance_m * 100.0  # Return in cm

def calculate_length_from_pixels(pixel_length, distance_cm, fx_px):
    """
    Calculate real-world length from pixel length at given distance
    Real_length = (pixel_length * distance) / fx
    """
    distance_m = distance_cm / 100.0
    real_length_m = (pixel_length * distance_m) / fx_px
    return real_length_m * 100.0  # Return in cm

def main():
    args = parse_args()
    left_path = Path(args.left)
    right_path = Path(args.right)
    baseline_cm = args.baseline_cm
    baseline_m = baseline_cm / 100.0
    fov_deg = args.fov_deg

    print(f"[INFO] Loading images...")
    left, right = load_images(left_path, right_path)
    h, w = left.shape[:2]
    
    # Calculate focal length from FOV
    fx = compute_fx_from_fov(w, fov_deg)
    print(f"[INFO] Image: {w}x{h}, FOV: {fov_deg}°, fx: {fx:.1f}px, Baseline: {baseline_cm}cm")

    # Create or load calibration
    calib = load_calib(args.calib)
    if calib is None:
        print("[INFO] Creating calibration from FOV...")
        save_calib_from_fov(args.calib, (w, h), baseline_m, fov_deg)
        calib = load_calib(args.calib)

    # Rectify images
    print("[INFO] Rectifying stereo pair...")
    left_rect, right_rect = rectify_pair(left, right, calib)
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # Compute disparity
    print("[INFO] Computing disparity map...")
    requested_num_disp = 16 * args.num_disp_mult
    requested_num_disp = _clamp_num_disp_to_width(requested_num_disp, w)
    disp = compute_disparity(left_gray, right_gray, num_disp=requested_num_disp, verbose=args.verbose)

    # Save disparity visualization
    vis_disp = np.zeros_like(left_gray, dtype=np.float32)
    finite = np.isfinite(disp)
    if np.any(finite):
        dmin = float(np.nanmin(disp[finite]))
        dmax = float(np.nanmax(disp[finite]))
        vis_disp[finite] = (disp[finite] - dmin) / (max(1e-9, dmax - dmin))
    cv2.imwrite(args.out_disp, (vis_disp * 255).astype(np.uint8))
    print(f"[INFO] Saved disparity to {args.out_disp}")

    # Detect fish
    bbox = None
    contour_for_vis = None
    
    if not args.no_yolo:
        bbox, yolo_err = try_yolo_bbox(left_rect)
        if bbox:
            print(f"[INFO] YOLO detected bbox: {bbox}")

    if bbox is None:
        res = detect_bbox_from_disp(disp, percentile=92)
        if res:
            bbox, contour_for_vis = res
            print(f"[INFO] Disparity-based bbox: {bbox}")

    if bbox is None:
        print("[ERROR] No fish detected")
        return

    # Find endpoints
    endpoints_cnt = endpoints_from_bbox(left_gray, bbox)
    if endpoints_cnt is None:
        print("[ERROR] Could not find endpoints")
        return
    
    (p1, p2), cnt = endpoints_cnt
    u1, v1 = int(p1[0]), int(p1[1])
    u2, v2 = int(p2[0]), int(p2[1])
    pixel_length = np.sqrt((u2-u1)**2 + (v2-v1)**2)
    print(f"[INFO] Endpoints: p1=({u1},{v1}), p2=({u2},{v2}), pixel_length={pixel_length:.1f}px")

    # Sample disparities
    samples = line_disp_samples(disp, (u1, v1), (u2, v2), n=201, win=11)
    robust_val = compute_trimmed_stat(samples, trim_percent=args.trim_percent)
    d1 = robust_disp_at(disp, u1, v1, max_win=41)
    d2 = robust_disp_at(disp, u2, v2, max_win=41)
    
    print(f"[INFO] Disparities: endpoint1={d1:.2f}px, endpoint2={d2:.2f}px, line_median={robust_val:.2f}px")

    # Choose disparity
    MIN_ACCEPTABLE = args.min_acceptable_disp_px
    chosen_disp = None
    if robust_val and robust_val > MIN_ACCEPTABLE and robust_val < MAX_REASONABLE_DISP:
        chosen_disp = robust_val
        reason = "line_median"
    elif d1 and d2 and d1 > MIN_ACCEPTABLE and d2 > MIN_ACCEPTABLE:
        if abs(d1 - d2) / max(d1, d2) < 0.4:
            chosen_disp = (d1 + d2) / 2.0
            reason = "endpoint_average"
    elif d1 and d1 > MIN_ACCEPTABLE:
        chosen_disp = d1
        reason = "endpoint1"
    elif d2 and d2 > MIN_ACCEPTABLE:
        chosen_disp = d2
        reason = "endpoint2"

    if chosen_disp is None:
        print(f"[ERROR] No valid disparity found")
        return

    # Calculate distance and length using corrected formulas
    distance_cm = calculate_distance_from_disparity(chosen_disp, baseline_cm, fx)
    fish_length_cm = calculate_length_from_pixels(pixel_length, distance_cm, fx)

    print(f"\n{'='*60}")
    print(f"[RESULT] Fish Measurement")
    print(f"{'='*60}")
    print(f"Disparity: {chosen_disp:.2f} px ({reason})")
    print(f"Distance to fish: {distance_cm:.1f} cm ({distance_cm/100:.2f} m)")
    print(f"Fish length: {fish_length_cm:.1f} cm")
    print(f"{'='*60}\n")

    # Visualize
    vis = left_rect.copy()
    x1,y1,x2,y2 = [int(v) for v in bbox]
    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
    if contour_for_vis is not None:
        cv2.drawContours(vis, [contour_for_vis], -1, (0,200,0), 2)
    cv2.circle(vis, (u1,v1), 8, (0,0,255), -1)
    cv2.circle(vis, (u2,v2), 8, (255,0,0), -1)
    cv2.line(vis, (u1,v1), (u2,v2), (255,255,0), 3)
    
    # Add text
    text = f"Length: {fish_length_cm:.1f} cm"
    dist_text = f"Distance: {distance_cm:.1f} cm"
    disp_text = f"Disparity: {chosen_disp:.1f} px"
    
    y_pos = max(20, y1 - 60)
    cv2.putText(vis, text, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(vis, dist_text, (x1, y_pos+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(vis, disp_text, (x1, y_pos+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    cv2.imwrite(args.out_vis, vis)
    print(f"[INFO] Saved result to {args.out_vis}")

if __name__ == "__main__":
    main()



# COMMMAND TO RUN:

# python3 stereo_fish_length.py \
#     --left image_test_left.png \
#     --right image_test_right.png \
#     --baseline-cm 1.0 \ 
#     --fov-deg 70.0 \ 
#     --verbose












































# YOLOv3 Real-time Stereo Fish Length Measurement

# #!/usr/bin/env python3
# """
# realtime_stereo_yolov3.py

# Real-time stereo vision system with YOLOv3 object detection for fish length measurement.

# Usage:
#     # Single image pair
#     python3 realtime_stereo_yolov3.py --left image_test_left.png --right image_test_right.png \
#         --baseline-cm 10.0 --fov-deg 70.0
    
#     # Real-time webcam (requires two cameras)
#     python3 realtime_stereo_yolov3.py --realtime --cam-left 0 --cam-right 1 \
#         --baseline-cm 10.0 --fov-deg 70.0
# """
# import argparse
# import os
# from pathlib import Path
# import numpy as np
# import cv2
# import math
# import time

# # ----------------- Configuration -----------------
# YOLOV3_WEIGHTS = "yolov3.weights"
# YOLOV3_CONFIG = "yolov3.cfg"
# YOLOV3_NAMES = "coco.names"

# SENSOR_NATIVE_W = 2592
# SENSOR_NATIVE_H = 1944
# PIXEL_SIZE_MM = 0.0014
# DEFAULT_BASELINE_CM = 10.0
# DEFAULT_FOV_DEG = 70.0
# NUM_DISP_MULT = 12
# MIN_ACCEPTABLE_DISP_PX = 4.0
# TRIM_PERCENT = 20
# MAX_REASONABLE_DISP = 1000.0

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--left", help="Left image path (for single image mode)")
#     p.add_argument("--right", help="Right image path (for single image mode)")
#     p.add_argument("--realtime", action="store_true", help="Enable real-time webcam mode")
#     p.add_argument("--cam-left", type=int, default=0, help="Left camera index")
#     p.add_argument("--cam-right", type=int, default=1, help="Right camera index")
#     p.add_argument("--baseline-cm", type=float, default=DEFAULT_BASELINE_CM)
#     p.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG)
#     p.add_argument("--yolo-weights", default=YOLOV3_WEIGHTS)
#     p.add_argument("--yolo-config", default=YOLOV3_CONFIG)
#     p.add_argument("--yolo-names", default=YOLOV3_NAMES)
#     p.add_argument("--confidence", type=float, default=0.5, help="YOLOv3 confidence threshold")
#     p.add_argument("--nms-threshold", type=float, default=0.4, help="Non-maximum suppression threshold")
#     p.add_argument("--target-class", default="fish", help="Target object class to detect")
#     p.add_argument("--out-vis", default="measurement_result.png")
#     p.add_argument("--out-disp", default="disparity_map.png")
#     p.add_argument("--verbose", action="store_true")
#     p.add_argument("--num-disp-mult", type=int, default=NUM_DISP_MULT)
#     p.add_argument("--show-fps", action="store_true", help="Display FPS in real-time mode")
#     return p.parse_args()

# class YOLOv3Detector:
#     """YOLOv3 object detector wrapper"""
    
#     def __init__(self, weights_path, config_path, names_path, confidence=0.5, nms_threshold=0.4):
#         self.confidence = confidence
#         self.nms_threshold = nms_threshold
        
#         # Load class names
#         if os.path.exists(names_path):
#             with open(names_path, 'r') as f:
#                 self.classes = [line.strip() for line in f.readlines()]
#         else:
#             # Default COCO classes including 'fish'
#             self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
#                            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
#                            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
#                            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
#                            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
#                            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#                            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#                            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#                            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
#                            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#                            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#                            'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#             print(f"[WARN] Names file not found, using default COCO classes")
        
#         # Load YOLOv3 network
#         try:
#             self.net = cv2.dnn.readNet(weights_path, config_path)
#             self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#             self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#             self.layer_names = self.net.getLayerNames()
#             self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
#             print(f"[INFO] YOLOv3 loaded successfully")
#         except Exception as e:
#             print(f"[ERROR] Failed to load YOLOv3: {e}")
#             print(f"[INFO] Download YOLOv3 files:")
#             print(f"  wget https://pjreddie.com/media/files/yolov3.weights")
#             print(f"  wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg")
#             print(f"  wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
#             raise
    
#     def detect(self, image, target_class=None):
#         """
#         Detect objects in image
#         Returns: list of (bbox, confidence, class_name) tuples
#         """
#         height, width = image.shape[:2]
        
#         # Create blob from image
#         blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
#         self.net.setInput(blob)
        
#         # Forward pass
#         outputs = self.net.forward(self.output_layers)
        
#         # Parse detections
#         boxes = []
#         confidences = []
#         class_ids = []
        
#         for output in outputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
                
#                 if confidence > self.confidence:
#                     # Object detected
#                     center_x = int(detection[0] * width)
#                     center_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)
                    
#                     # Rectangle coordinates
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)
                    
#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)
        
#         # Non-maximum suppression
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.nms_threshold)
        
#         detections = []
#         if len(indices) > 0:
#             for i in indices.flatten():
#                 x, y, w, h = boxes[i]
#                 class_name = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                
#                 # Filter by target class if specified
#                 if target_class is None or class_name.lower() == target_class.lower():
#                     bbox = (x, y, x + w, y + h)
#                     detections.append((bbox, confidences[i], class_name))
        
#         return detections

# def compute_fx_from_fov(img_w, fov_deg):
#     """Calculate focal length in pixels from field of view"""
#     fov_rad = math.radians(fov_deg)
#     fx = img_w / (2 * math.tan(fov_rad / 2))
#     return fx

# def load_images(left_path, right_path):
#     left = cv2.imread(str(left_path))
#     right = cv2.imread(str(right_path))
#     if left is None or right is None:
#         raise FileNotFoundError(f"Images not found: {left_path}, {right_path}")
    
#     # Auto-resize if dimensions don't match
#     if left.shape != right.shape:
#         print(f"[WARN] Resizing images to match dimensions")
#         h_left, w_left = left.shape[:2]
#         h_right, w_right = right.shape[:2]
#         target_h = min(h_left, h_right)
#         target_w = min(w_left, w_right)
#         left = cv2.resize(left, (target_w, target_h), interpolation=cv2.INTER_AREA)
#         right = cv2.resize(right, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
#     return left, right

# def _clamp_num_disp_to_width(num_disp, img_w):
#     if num_disp <= 0:
#         num_disp = 16
#     if num_disp % 16 != 0:
#         num_disp = (num_disp // 16 + 1) * 16
#     max_allowed = max(16, (img_w // 16) * 16)
#     return min(int(num_disp), max_allowed)

# def compute_disparity(left_gray, right_gray, num_disp=None):
#     """Compute disparity map from stereo pair"""
#     window_size = 7
#     min_disp = 0
#     h, w = left_gray.shape[:2]
    
#     if num_disp is None:
#         num_disp = 16 * NUM_DISP_MULT
#     num_disp = _clamp_num_disp_to_width(num_disp, w)
    
#     stereo = cv2.StereoSGBM_create(
#         minDisparity=min_disp,
#         numDisparities=num_disp,
#         blockSize=window_size,
#         P1=8*3*window_size**2,
#         P2=32*3*window_size**2,
#         disp12MaxDiff=2,
#         uniquenessRatio=5,
#         speckleWindowSize=200,
#         speckleRange=16,
#         preFilterCap=63,
#         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#     )
    
#     disp_raw = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
#     disp = disp_raw.copy()
#     disp[(disp <= 0) | (~np.isfinite(disp)) | (disp > MAX_REASONABLE_DISP)] = np.nan
    
#     # Post-processing
#     disp_smooth = cv2.medianBlur(np.nan_to_num(disp, nan=0.0).astype(np.float32), 5)
#     disp_smooth = disp_smooth.astype(np.float32)
#     disp_smooth[disp_smooth == 0] = np.nan
    
#     # Remove speckles
#     finite_mask = np.isfinite(disp_smooth).astype(np.uint8)*255
#     kernel = np.ones((5,5), np.uint8)
#     finite_mask = cv2.morphologyEx(finite_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     disp_smooth[finite_mask==0] = np.nan
    
#     return disp_smooth

# def endpoints_from_bbox(gray, bbox):
#     """Find object endpoints from bounding box"""
#     x1,y1,x2,y2 = [int(v) for v in bbox]
#     crop = gray[y1:y2, x1:x2]
#     if crop.size == 0:
#         return None
    
#     blur = cv2.GaussianBlur(crop, (5,5), 0)
#     _, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     kernel = np.ones((3,3), np.uint8)
#     th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
#     contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if not contours:
#         # Fallback to bbox corners
#         return ((x1, (y1+y2)//2), (x2, (y1+y2)//2)), None
    
#     cnt = max(contours, key=cv2.contourArea)
#     hull = cv2.convexHull(cnt)
    
#     # Find longest pair in hull
#     pts = hull.reshape(-1,2)
#     if len(pts) < 2:
#         return ((x1, (y1+y2)//2), (x2, (y1+y2)//2)), None
    
#     maxd = 0.0
#     p1 = p2 = None
#     for i in range(len(pts)):
#         for j in range(i+1, len(pts)):
#             d = np.sum((pts[i] - pts[j])**2)
#             if d > maxd:
#                 maxd = d
#                 p1 = tuple(pts[i])
#                 p2 = tuple(pts[j])
    
#     p1 = (int(p1[0] + x1), int(p1[1] + y1))
#     p2 = (int(p2[0] + x1), int(p2[1] + y1))
#     return (p1, p2), cnt

# def robust_disp_at(disp, u, v, max_win=21):
#     """Get robust disparity estimate at point"""
#     h,w = disp.shape
#     u_c = np.clip(int(round(u)), 0, w-1)
#     v_c = np.clip(int(round(v)), 0, h-1)
#     for win in range(3, max_win+1, 2):
#         x0 = max(0, u_c - win); x1 = min(w, u_c + win + 1)
#         y0 = max(0, v_c - win); y1 = min(h, v_c + win + 1)
#         patch = disp[y0:y1, x0:x1]
#         patch = patch[np.isfinite(patch) & (patch > 0)]
#         if patch.size > 0:
#             return float(np.median(patch))
#     return None

# def line_disp_samples(disp, p1, p2, n=101, win=11):
#     """Sample disparities along line"""
#     samples = []
#     (u1,v1) = p1; (u2,v2) = p2
#     for i in range(n):
#         t = i / max(1, n-1)
#         u = u1 * (1 - t) + u2 * t
#         v = v1 * (1 - t) + v2 * t
#         d = robust_disp_at(disp, u, v, max_win=win)
#         if d is not None and d > 0:
#             samples.append(d)
#     return samples

# def compute_trimmed_stat(samples, trim_percent=TRIM_PERCENT):
#     """Compute trimmed median"""
#     if len(samples) == 0:
#         return None
#     arr = np.sort(np.asarray(samples, dtype=np.float64))
#     k = int(len(arr) * (trim_percent/100.0))
#     if k*2 >= len(arr):
#         return float(np.median(arr))
#     trimmed = arr[k:len(arr)-k]
#     if trimmed.size == 0:
#         return float(np.median(arr))
#     return float(np.median(trimmed))

# def calculate_distance_from_disparity(disparity_px, baseline_cm, fx_px):
#     """Calculate distance: Z = (fx * baseline) / disparity"""
#     if disparity_px <= 0:
#         return None
#     baseline_m = baseline_cm / 100.0
#     distance_m = (fx_px * baseline_m) / disparity_px
#     return distance_m * 100.0  # cm

# def calculate_length_from_pixels(pixel_length, distance_cm, fx_px):
#     """Calculate real-world length: length = (pixels * distance) / fx"""
#     distance_m = distance_cm / 100.0
#     real_length_m = (pixel_length * distance_m) / fx_px
#     return real_length_m * 100.0  # cm

# def process_frame_pair(left, right, detector, args, fx):
#     """Process a stereo frame pair and return measurements"""
#     h, w = left.shape[:2]
    
#     # Convert to grayscale for disparity
#     left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
#     right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
#     # Compute disparity
#     requested_num_disp = 16 * args.num_disp_mult
#     requested_num_disp = _clamp_num_disp_to_width(requested_num_disp, w)
#     disp = compute_disparity(left_gray, right_gray, num_disp=requested_num_disp)
    
#     # Detect objects with YOLOv3
#     detections = detector.detect(left, target_class=args.target_class)
    
#     if not detections:
#         return left, None, "No objects detected"
    
#     # Use best detection (highest confidence)
#     detections.sort(key=lambda x: x[1], reverse=True)
#     bbox, confidence, class_name = detections[0]
    
#     # Find endpoints
#     endpoints_result = endpoints_from_bbox(left_gray, bbox)
#     if endpoints_result is None:
#         return left, None, f"Found {class_name} but couldn't measure"
    
#     (p1, p2), cnt = endpoints_result
#     u1, v1 = int(p1[0]), int(p1[1])
#     u2, v2 = int(p2[0]), int(p2[1])
#     pixel_length = np.sqrt((u2-u1)**2 + (v2-v1)**2)
    
#     # Sample disparities
#     samples = line_disp_samples(disp, (u1, v1), (u2, v2), n=201, win=11)
#     robust_val = compute_trimmed_stat(samples, trim_percent=args.trim_percent)
#     d1 = robust_disp_at(disp, u1, v1, max_win=41)
#     d2 = robust_disp_at(disp, u2, v2, max_win=41)
    
#     # Choose disparity
#     chosen_disp = None
#     reason = ""
#     MIN_ACCEPTABLE = MIN_ACCEPTABLE_DISP_PX
    
#     if robust_val and robust_val > MIN_ACCEPTABLE and robust_val < MAX_REASONABLE_DISP:
#         chosen_disp = robust_val
#         reason = "line_median"
#     elif d1 and d2 and d1 > MIN_ACCEPTABLE and d2 > MIN_ACCEPTABLE:
#         if abs(d1 - d2) / max(d1, d2) < 0.4:
#             chosen_disp = (d1 + d2) / 2.0
#             reason = "endpoint_avg"
    
#     if chosen_disp is None:
#         return left, None, f"Found {class_name} but disparity invalid"
    
#     # Calculate measurements
#     distance_cm = calculate_distance_from_disparity(chosen_disp, args.baseline_cm, fx)
#     length_cm = calculate_length_from_pixels(pixel_length, distance_cm, fx)
    
#     # Visualize
#     vis = left.copy()
#     x1, y1, x2, y2 = bbox
#     cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.circle(vis, (u1, v1), 8, (0, 0, 255), -1)
#     cv2.circle(vis, (u2, v2), 8, (255, 0, 0), -1)
#     cv2.line(vis, (u1, v1), (u2, v2), (255, 255, 0), 3)
    
#     # Add text
#     info_text = [
#         f"{class_name.upper()} ({confidence:.2f})",
#         f"Length: {length_cm:.1f} cm",
#         f"Distance: {distance_cm:.1f} cm",
#         f"Disparity: {chosen_disp:.1f}px"
#     ]
    
#     y_offset = max(20, y1 - 80)
#     for i, text in enumerate(info_text):
#         color = (0, 255, 0) if i == 1 else (255, 255, 255)
#         thickness = 2 if i == 1 else 1
#         cv2.putText(vis, text, (x1, y_offset + i*25), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
    
#     result = {
#         'class': class_name,
#         'confidence': confidence,
#         'length_cm': length_cm,
#         'distance_cm': distance_cm,
#         'disparity': chosen_disp,
#         'bbox': bbox
#     }
    
#     return vis, result, None

# def realtime_mode(args, detector, fx):
#     """Real-time processing from webcams"""
#     print(f"[INFO] Opening cameras {args.cam_left} and {args.cam_right}...")
    
#     cap_left = cv2.VideoCapture(args.cam_left)
#     cap_right = cv2.VideoCapture(args.cam_right)
    
#     if not cap_left.isOpened() or not cap_right.isOpened():
#         print("[ERROR] Could not open cameras")
#         return
    
#     print("[INFO] Starting real-time measurement. Press 'q' to quit, 's' to save frame")
    
#     frame_count = 0
#     fps_time = time.time()
#     fps = 0
    
#     while True:
#         ret_left, left = cap_left.read()
#         ret_right, right = cap_right.read()
        
#         if not ret_left or not ret_right:
#             print("[WARN] Failed to grab frames")
#             break
        
#         # Resize if needed
#         if left.shape != right.shape:
#             h = min(left.shape[0], right.shape[0])
#             w = min(left.shape[1], right.shape[1])
#             left = cv2.resize(left, (w, h))
#             right = cv2.resize(right, (w, h))
        
#         # Process frame
#         vis, result, error = process_frame_pair(left, right, detector, args, fx)
        
#         # Calculate FPS
#         frame_count += 1
#         if frame_count % 10 == 0:
#             current_time = time.time()
#             fps = 10 / (current_time - fps_time)
#             fps_time = current_time
        
#         # Display FPS
#         if args.show_fps:
#             cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
#         # Display result or error
#         if error:
#             cv2.putText(vis, error, (10, vis.shape[0] - 20),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         cv2.imshow('Stereo Measurement', vis)
        
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('s') and result:
#             timestamp = int(time.time())
#             filename = f"measurement_{timestamp}.png"
#             cv2.imwrite(filename, vis)
#             print(f"[INFO] Saved to {filename}")
    
#     cap_left.release()
#     cap_right.release()
#     cv2.destroyAllWindows()

# def single_image_mode(args, detector, fx):
#     """Process a single stereo image pair"""
#     print(f"[INFO] Loading images...")
#     left, right = load_images(args.left, args.right)
    
#     print(f"[INFO] Processing...")
#     vis, result, error = process_frame_pair(left, right, detector, args, fx)
    
#     if error:
#         print(f"[ERROR] {error}")
#     elif result:
#         print(f"\n{'='*60}")
#         print(f"[RESULT] Measurement")
#         print(f"{'='*60}")
#         print(f"Object: {result['class']} (confidence: {result['confidence']:.2f})")
#         print(f"Length: {result['length_cm']:.1f} cm")
#         print(f"Distance: {result['distance_cm']:.1f} cm")
#         print(f"Disparity: {result['disparity']:.2f} px")
#         print(f"{'='*60}\n")
    
#     cv2.imwrite(args.out_vis, vis)
#     print(f"[INFO] Saved result to {args.out_vis}")
    
#     # Display
#     cv2.imshow('Result', vis)
#     print("[INFO] Press any key to close...")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def main():
#     args = parse_args()
    
#     # Validate mode
#     if args.realtime:
#         print("[INFO] Real-time mode enabled")
#     elif not args.left or not args.right:
#         print("[ERROR] Either --realtime or both --left and --right must be specified")
#         return
    
#     # Initialize YOLOv3
#     print("[INFO] Loading YOLOv3...")
#     try:
#         detector = YOLOv3Detector(
#             args.yolo_weights, 
#             args.yolo_config, 
#             args.yolo_names,
#             confidence=args.confidence,
#             nms_threshold=args.nms_threshold
#         )
#     except Exception as e:
#         print(f"[ERROR] Failed to load YOLOv3: {e}")
#         return
    
#     # Calculate focal length
#     if args.realtime:
#         # Use default resolution for webcams
#         img_w = 640
#     else:
#         left, _ = load_images(args.left, args.right)
#         img_w = left.shape[1]
    
#     fx = compute_fx_from_fov(img_w, args.fov_deg)
#     print(f"[INFO] FOV: {args.fov_deg}°, fx: {fx:.1f}px, Baseline: {args.baseline_cm}cm")
    
#     # Run appropriate mode
#     if args.realtime:
#         realtime_mode(args, detector, fx)
#     else:
#         single_image_mode(args, detector, fx)

# if __name__ == "__main__":
#     main()