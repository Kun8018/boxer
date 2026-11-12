#! /usr/bin/env python3
"""RQ1 face anchors overlaid on the TrackerViewer layout (hohen Aria demo).

This matches https://facebookresearch.github.io/boxer/images/hohen_demo.mp4:
the window shows "Per-Frame Detections" (2D boxes), "Tracked 3DBBs" (tracked
OBBs in the photo) and a 3D/top-down viewport. The RQ1 annotations (yellow
anchor cross + red velocity arrow + speed/distance label) are drawn INSIDE the
"Tracked 3DBBs" panel, mapped onto its exact image rect - no separate view.

Prerequisites:
    bash scripts/download_ckpts.sh
    bash scripts/download_aria_data.sh hohen_gen1
    python run_boxer.py --input hohen_gen1 --track --skip_viz --max_n 90

Run:
    python demo_face_anchors_real.py
    python demo_face_anchors_real.py --record --record_fps 9   # save MP4
"""

import argparse
import os

import numpy as np
import torch

import utils.imgui_compat as imgui
from demo_face_anchors import FACE_ORDER, GeomBuilder, face_anchor
from utils.file_io import read_obb_csv
from utils.tw.obb import ObbTW
from utils.viewer_3d import (
    TrackerViewer,
    add_common_args,
    build_seq_ctx,
    launch_viewer,
    load_common,
    subsample_timed_obbs,
)


def stitch_tracks(timed_obbs, max_center_gap=0.5):
    """Greedy nearest-center linking across consecutive frames.

    Returns list of chains, each a list of (ts, obb_row).
    """
    timestamps = sorted(timed_obbs.keys())
    chains = []
    active = {}  # row idx in current frame -> chain index
    for ts in timestamps:
        obbs = timed_obbs[ts]
        centers = [obbs[i].bb3_center_world.reshape(3).double().numpy()
                   for i in range(len(obbs))]
        tails = [(ci, chains[ci][-1][1].bb3_center_world.reshape(3).double().numpy())
                 for ci in set(active.values())]
        candidates = []
        for ci, pc in tails:
            for i, c in enumerate(centers):
                d = float(np.linalg.norm(pc - c))
                if d < max_center_gap:
                    candidates.append((d, ci, i))
        candidates.sort()
        used_chain, used_box = set(), set()
        next_active = {}
        for _d, ci, i in candidates:
            if ci in used_chain or i in used_box:
                continue
            used_chain.add(ci)
            used_box.add(i)
            chains[ci].append((ts, obbs[i]))
            next_active[i] = ci
        for i in range(len(obbs)):
            if i not in used_box:
                ci = len(chains)
                chains.append([(ts, obbs[i])])
                next_active[i] = ci
        active = next_active
    return chains


def build_kinematics(timed_obbs):
    """Stitched tracks -> per-sample velocity + accumulated distance.

    Returns dict: ts -> list of (center_xyz np.ndarray, velocity np.ndarray,
    distance float) for quick per-frame lookup.
    """
    per_frame = {}
    for entries in stitch_tracks(timed_obbs):
        entries.sort(key=lambda e: e[0])
        samples = []
        for ts, row in entries:
            samples.append(dict(
                ts=int(ts),
                center=row.bb3_center_world.reshape(3).double().numpy(),
                size=row.bb3_diagonal.reshape(3).double().numpy(),
                R=row.T_world_object.R.reshape(3, 3).double().numpy(),
            ))
        n = len(samples)
        dist = 0.0
        for j, s in enumerate(samples):
            if 0 < j < n - 1:
                dt = (samples[j + 1]["ts"] - samples[j - 1]["ts"]) / 1e9
                vel = (samples[j + 1]["center"] - samples[j - 1]["center"]) \
                    / max(dt, 1e-6)
            elif j == 0 and n > 1:
                dt = (samples[1]["ts"] - samples[0]["ts"]) / 1e9
                vel = (samples[1]["center"] - samples[0]["center"]) / max(dt, 1e-6)
            elif j == n - 1 and n > 1:
                dt = (samples[j]["ts"] - samples[j - 1]["ts"]) / 1e9
                vel = (samples[j]["center"] - samples[j - 1]["center"]) \
                    / max(dt, 1e-6)
            else:
                vel = np.zeros(3)
            if j > 0:
                dist += float(np.linalg.norm(
                    samples[j]["center"] - samples[j - 1]["center"]))
            per_frame.setdefault(s["ts"], []).append(
                (s["center"], s["size"], s["R"], vel, dist))
    return per_frame


def main():
    parser = argparse.ArgumentParser(
        description="RQ1 face anchors overlaid in TrackerViewer (hohen)")
    add_common_args(parser)
    parser.set_defaults(input="hohen_gen1")
    parser.add_argument("--max_speed", type=float, default=2.0,
                        help="m/s mapped to full arrow length")
    parser.add_argument("--record", action="store_true",
                        help="Auto-play once and record the window to an MP4")
    parser.add_argument("--record_fps", type=float, default=9.0,
                        help="Playback/recording frames per second")
    args = parser.parse_args()

    input_path, dataset_type, seq_name, log_dir, view_path, load_view_data = (
        load_common(args)
    )

    csv_path = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    if not os.path.exists(csv_path):
        raise IOError(
            f"3D BB CSV not found: {csv_path}\n"
            "Run the pipeline first, e.g.\n"
            f"  python run_boxer.py --input {seq_name} --track --skip_viz")

    timed_obbs = read_obb_csv(csv_path)
    timed_obbs = subsample_timed_obbs(
        timed_obbs, skip_n=args.skip_n, start_n=args.start_n, max_n=args.max_n
    )
    print(f"Loaded {len(timed_obbs)} frames from {os.path.basename(csv_path)}")

    kinematics = build_kinematics(timed_obbs)
    seq_ctx = build_seq_ctx(input_path, dataset_type)

    class FaceAnchorTrackerViewer(TrackerViewer):
        title = "RQ1 face anchors (hohen TrackerViewer)"

        def __init__(self, **kw):
            self.active_face = "top"
            self.show_all_faces = False
            self.show_anchors = True
            self.show_anchor_labels = True
            self.max_speed = args.max_speed
            self._tracked_image_rect = None
            self._record_enabled = args.record
            self._record_fps = args.record_fps
            self._record_last_idx = -1
            self._record_frames_dir = os.path.join(
                log_dir, "rq1_face_anchors_video")
            self._record_mp4 = os.path.join(log_dir, "rq1_face_anchors_demo.mp4")
            if self._record_enabled:
                os.makedirs(self._record_frames_dir, exist_ok=True)
                for f in os.listdir(self._record_frames_dir):
                    os.remove(os.path.join(self._record_frames_dir, f))

            super().__init__(
                timed_obbs=timed_obbs,
                root_path=log_dir,
                seq_ctx=seq_ctx,
                load_view_data=load_view_data,
                view_save_path=view_path,
                **kw,
            )
            self.playback_fps = self._record_fps

        # -- hooks -----------------------------------------------------

        def on_render(self, time, frame_time):
            if (self._record_enabled and not self.is_playing
                    and self.current_frame_idx == 0
                    and self._record_last_idx == -1):
                import time as _t
                self.is_playing = True
                self.follow_view = True
                self._last_step_time = _t.time()

            super().on_render(time, frame_time)

            if not self._record_enabled:
                return
            if self.current_frame_idx != self._record_last_idx:
                from PIL import Image
                self._record_last_idx = self.current_frame_idx
                w, h = self.wnd.size
                fb_w, fb_h = self.ctx.fbo.size
                data = self.ctx.fbo.read(
                    components=3, alignment=1,
                    viewport=(0, fb_h - h, w, h))
                img = Image.frombytes("RGB", (w, h), data)
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                img.save(os.path.join(
                    self._record_frames_dir,
                    f"f_{self.current_frame_idx:04d}.png"))
            if self.current_frame_idx >= self.total_frames - 1:
                if self.is_playing:
                    self.is_playing = False
                else:
                    self._encode_recording()
                    self.wnd.close()

        def _encode_recording(self):
            import glob
            import subprocess
            import imageio_ffmpeg
            frames = sorted(glob.glob(os.path.join(
                self._record_frames_dir, "f_*.png")))
            if not frames:
                return
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = (
                f"{ffmpeg} -hide_banner -loglevel error -y "
                f"-framerate {self._record_fps} -pattern_type glob "
                f"-i '{self._record_frames_dir}/f_*.png' "
                f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "
                f"-c:v libx264 -preset slow -crf 18 "
                f"-pix_fmt yuv420p -movflags +faststart {self._record_mp4}"
            )
            print("[REC] encoding:", self._record_mp4)
            subprocess.run(cmd, shell=True, check=True)
            print(f"[REC] saved {len(frames)} frames -> {self._record_mp4}")

        # -- overlay inside the real "Tracked 3DBBs" panel -------------

        def render_ui(self):
            # While the base draws, capture every image rect. The
            # "Tracked 3DBBs" photo is the last one drawn each frame.
            captured = []
            orig_image = imgui.image

            def _capture(tex_id, draw_w, draw_h, *a, **k):
                result = orig_image(tex_id, draw_w, draw_h, *a, **k)
                rect_min = imgui.get_item_rect_min()
                tex_w, tex_h = self._rgb_tex_size
                captured.append((rect_min.x, rect_min.y,
                                 draw_w, draw_h, tex_w, tex_h))
                return result

            imgui.image = _capture
            try:
                super().render_ui()
            finally:
                imgui.image = orig_image

            self._tracked_image_rect = captured[-1] if captured else None

            imgui.begin("RQ1 face anchors")
            current = FACE_ORDER.index(self.active_face)
            changed, current = imgui.combo(
                "Anchor face (IV1)", current,
                [f.capitalize() for f in FACE_ORDER])
            if changed:
                self.active_face = FACE_ORDER[current]
            _c, self.show_all_faces = imgui.checkbox(
                "All six faces (debug)", self.show_all_faces)
            _c, self.show_anchors = imgui.checkbox(
                "Show annotations", self.show_anchors)
            _c, self.show_anchor_labels = imgui.checkbox(
                "Speed/distance labels", self.show_anchor_labels)
            _c, self.max_speed = imgui.slider_float(
                "Speed at full arrow (m/s)", self.max_speed, 0.2, 5.0)
            imgui.end()

            if not (self.show_anchors and self._tracked_image_rect):
                return

            ts = self.sorted_timestamps[self.current_frame_idx]
            nav_ts = self._get_navigation_timestamp(
                self.current_frame_idx, ts)
            cam, T_wr = self._get_cam_and_pose(nav_ts)
            if cam is None or T_wr is None:
                return

            # Build anchor world geometry for every tracked instance, using
            # the stitched velocity matched by nearest center.
            kin = kinematics.get(ts, [])
            lb = GeomBuilder()
            faces = FACE_ORDER if self.show_all_faces else [self.active_face]
            label_points = []
            name_points = []
            active_tracks = self.tracker._get_active_tracks()
            for track in active_tracks:
                obb = track.obb
                center = obb.bb3_center_world.reshape(3).double().numpy()
                size = obb.bb3_diagonal.reshape(3).double().numpy()
                R = obb.T_world_object.R.reshape(3, 3).double().numpy()
                # nearest kinematic sample at this frame
                best, best_d = None, 0.3
                for kc, ks_, kr_, kv, kd in kin:
                    d = float(np.linalg.norm(kc - center))
                    if d < best_d:
                        best, best_d = (kv, kd), d
                for face in faces:
                    anchor, normal_w, ta, tb = face_anchor(
                        center, size, R, face)
                    lb.cross_mark(anchor, ta, tb, (1.0, 0.80, 0.15))
                    vel = best[0] if best is not None else np.zeros(3)
                    speed = float(np.linalg.norm(vel))
                    frac = min(1.0, speed / max(self.max_speed, 1e-6))
                    direction = vel if speed > 1e-3 else np.array([0, 0, 1.0])
                    lb.arrow(anchor + normal_w * 0.05, direction,
                             (0.95, 0.25, 0.20),
                             length=0.20 + frac * 0.85)
                    dist_txt = f"{best[1]:.1f}" if best is not None else "-"
                    label_points.append((
                        anchor + normal_w * 0.28,
                        f"{speed:.2f} m/s | d {dist_txt} m"))
                    # Object name printed ON the anchor surface, identifying
                    # what the box is.
                    name = str(getattr(track, "cached_text", "?")).strip()
                    if name and name != "?":
                        name_points.append((
                            anchor + normal_w * 0.02, name))

            # Project segments with the same camera used by the base overlays.
            img_x, img_y, draw_w, draw_h, tex_w, tex_h = \
                self._tracked_image_rect
            sx = draw_w / tex_w * self._rgb_img_scale
            sy = draw_h / tex_h * self._rgb_img_scale

            segs_world = [p for s in lb.segs for p in (s[0], s[1])]
            if segs_world:
                pts = torch.as_tensor(
                    np.asarray(segs_world), dtype=torch.float32)
                T_world_cam = T_wr @ cam.T_camera_rig.inverse()
                pts_cam = T_world_cam.inverse().transform(pts)
                proj_cam = cam
                if self._rgb_vrs_w > 0 and self._rgb_vrs_h > 0:
                    cw = cam.size[..., 0].item()
                    ch = cam.size[..., 1].item()
                    if abs(cw - self._rgb_vrs_w) > 1 or abs(ch - self._rgb_vrs_h) > 1:
                        proj_cam = cam.scale_to_size(
                            (self._rgb_vrs_w, self._rgb_vrs_h))
                p2d, valid = proj_cam.project(
                    pts_cam.unsqueeze(0),
                    fov_deg=140.0 if self._vrs_is_nebula else 120.0)
                p2d = p2d.squeeze(0).cpu().numpy()
                valid = valid.squeeze(0).cpu().numpy()
                in_front = pts_cam[:, 2].cpu().numpy() > 0.1

                dl = imgui.get_foreground_draw_list()
                for i, (p0, p1, c) in enumerate(lb.segs):
                    if not (valid[2 * i] and valid[2 * i + 1]
                            and in_front[2 * i] and in_front[2 * i + 1]):
                        continue
                    a = p2d[2 * i]
                    b = p2d[2 * i + 1]
                    col = imgui.get_color_u32_rgba(
                        float(c[0]), float(c[1]), float(c[2]), 1.0)
                    dl.add_line(img_x + a[0] * sx, img_y + a[1] * sy,
                                img_x + b[0] * sx, img_y + b[1] * sy,
                                col, 3.0)
                if self.show_anchor_labels:
                    lp = [p for p, _ in label_points]
                    lp_t = torch.as_tensor(np.asarray(lp), dtype=torch.float32)
                    lc = T_world_cam.inverse().transform(lp_t)
                    l2d, lv = proj_cam.project(
                        lc.unsqueeze(0),
                        fov_deg=140.0 if self._vrs_is_nebula else 120.0)
                    l2d = l2d.squeeze(0).cpu().numpy()
                    lv = lv.squeeze(0).cpu().numpy()
                    col = imgui.get_color_u32_rgba(1.0, 0.95, 0.75, 1.0)
                    for k, (_, text) in enumerate(label_points):
                        if lv[k] and lc[k, 2].item() > 0.1:
                            dl.add_text(img_x + l2d[k][0] * sx + 8,
                                        img_y + l2d[k][1] * sy - 8,
                                        col, text)

                # Object name ON the anchor surface: bold yellow text with a
                # dark backing rect for readability.
                if name_points:
                    np_pts = [p for p, _ in name_points]
                    np_t = torch.as_tensor(
                        np.asarray(np_pts), dtype=torch.float32)
                    nc_pts = T_world_cam.inverse().transform(np_t)
                    n2d, nv2 = proj_cam.project(
                        nc_pts.unsqueeze(0),
                        fov_deg=140.0 if self._vrs_is_nebula else 120.0)
                    n2d = n2d.squeeze(0).cpu().numpy()
                    nv2 = nv2.squeeze(0).cpu().numpy()
                    name_col = imgui.get_color_u32_rgba(
                        1.0, 0.85, 0.2, 1.0)
                    bg_col = imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.65)
                    for k, (_, text) in enumerate(name_points):
                        if nv2[k] and nc_pts[k, 2].item() > 0.1:
                            tx = img_x + n2d[k][0] * sx
                            ty = img_y + n2d[k][1] * sy
                            tw, th = imgui.calc_text_size(text)
                            dl.add_rect_filled(
                                tx - 3, ty - 2,
                                tx + tw + 3, ty + th + 2, bg_col)
                            dl.add_text(tx, ty, name_col, text)

    launch_viewer(FaceAnchorTrackerViewer)


if __name__ == "__main__":
    main()
