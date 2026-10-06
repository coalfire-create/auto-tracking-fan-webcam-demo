#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 자동 추적 선풍기 - 웹캠 추적 시뮬레이션 (공모전 시연용)

노트북 웹캠으로 실제 사람을 인식하고, Raspberry Pi 에 올릴 것과 같은
Pan/Tilt 제어 로직(PID + 가감속 제한 + 탐색 + 릴레이 자동 OFF)으로
'가상 선풍기'가 사람을 따라가는 모습을 화면에 보여 줍니다.

설치:   pip install opencv-python numpy
        (선택, 더 정확한 인식) pip install ultralytics   -> YOLOv8n 자동 사용
실행:   python webcam_tracking_demo.py            (기본 웹캠)
        python webcam_tracking_demo.py --video 영상.mp4
        python webcam_tracking_demo.py --detector hog   (YOLO 없이)

사람/마네킹 구분: liveness.py (같은 폴더에 두세요)
        · 미세 움직임(기본) + 열화상 MLX90640(라즈베리파이에 연결 시 자동 사용)
        · 사진·포스터·마네킹은 'MANNEQUIN'(빨강 X)으로 표시되고 추적·선풍기 ON 대상에서 제외

조작 키:  A 자동 추적 / M 수동 / ←→↑↓ 수동 이동(수동 모드) / P 선풍기 전원 / L 사람·마네킹 구분 ON/OFF / R 지표 초기화 / Q 종료
로그:    실행 폴더에 tracking_log_YYYYmmdd_HHMMSS.csv 저장 (그래프·보고서용 데이터)
"""
import argparse, csv, math, time, datetime, random
import numpy as np
import cv2
from liveness import LivenessJudge, ThermalSensor, draw_tracks

# ---------------- 설정 (시뮬레이터 기본값과 동일) ----------------
HFOV, VFOV = 66.0, 41.0          # Camera Module 3 (Standard) 시야각
CFG = dict(kp=5.0, ki=0.5, kd=0.15, vmax=90.0, amax=200.0, dead=1.0,   # v2: GT2 벨트 1:3 감속 기준
           pan_min=-90.0, pan_max=90.0, tilt_min=-30.0, tilt_max=45.0,
           tilt_kp=5.0, tilt_vmax=60.0, micro=16, belt=3, off_delay=8.0, search_after=1.5)


class PanTiltController:
    """NEMA17(Pan) 속도·가속 제한 PID + 서보(Tilt) 위치 제어. Pi 코드에 그대로 옮겨 쓸 수 있는 구조."""
    def __init__(self, c):
        self.c = c; self.pan = 0.0; self.pan_v = 0.0; self.tilt = -5.0
        self.tgt_pan = 0.0; self.tgt_tilt = -5.0; self.I = 0.0

    def step(self, dt, vmax=None):
        c = self.c; vmax = vmax or c['vmax']
        err = self.tgt_pan - self.pan
        e = 0.0 if abs(err) < c['dead'] else err
        self.I = float(np.clip(self.I + e * dt, -15, 15))
        v = c['kp'] * e + c['ki'] * self.I - c['kd'] * self.pan_v
        v = float(np.clip(v, -vmax, vmax))
        vs = math.sqrt(2 * c['amax'] * abs(err)) * 0.95          # 정지거리 제한 (가감속 프로파일)
        v = float(np.clip(v, -vs, vs))
        self.pan_v += float(np.clip(v - self.pan_v, -c['amax'] * dt, c['amax'] * dt))
        self.pan = float(np.clip(self.pan + self.pan_v * dt, c['pan_min'], c['pan_max']))
        te = float(np.clip(self.tgt_tilt, c['tilt_min'], c['tilt_max'])) - self.tilt
        self.tilt += float(np.clip(c['tilt_kp'] * te, -c['tilt_vmax'], c['tilt_vmax'])) * dt

    @property
    def steps(self):  # 1.8° 스테퍼, 1/16 마이크로스텝, 벨트 1:3
        return int(round(self.pan / (1.8 / self.c['micro']) * self.c['belt']))


class Detector:
    def __init__(self, kind, model='yolov8n.pt', ncnn=False):
        self.kind = None; self.label = 'HOG'
        if kind in ('auto', 'yolo'):
            try:
                import os
                from ultralytics import YOLO
                self.model = YOLO(model)
                name = os.path.splitext(os.path.basename(model))[0]
                if ncnn:  # 라즈베리파이: NCNN 형식이 PyTorch보다 몇 배 빠름 (처음 1회만 변환)
                    ncnn_dir = os.path.join(os.path.dirname(os.path.abspath(model)), name + '_ncnn_model')
                    if not os.path.isdir(ncnn_dir):
                        print('[안내] NCNN 형식으로 변환 중(처음 1회)...')
                        ncnn_dir = self.model.export(format='ncnn', imgsz=416)
                    self.model = YOLO(ncnn_dir, task='detect'); name += ' NCNN'
                self.kind = 'yolo'; self.label = ('YOLO' + name[4:]) if name.lower().startswith('yolo') else name
            except Exception as ex:
                if kind == 'yolo': raise
                print('[안내] ultralytics 가 없어 OpenCV HOG 사람 인식으로 실행합니다:', ex)
        if self.kind is None:
            self.hog = cv2.HOGDescriptor(); self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.kind = 'hog'

    def __call__(self, frame):
        """화면의 모든 'person' 박스 [(x0,y0,x1,y1,conf), ...] — 마네킹·사진도 여기서는 사람으로 나옴"""
        h, w = frame.shape[:2]
        if self.kind == 'yolo':
            r = self.model(frame, classes=[0], verbose=False, imgsz=416)[0]
            return [tuple(b.xyxy[0].tolist()) + (float(b.conf[0]),) for b in r.boxes if float(b.conf[0]) >= 0.4]
        # HOG (전신) + 얼굴 검출로 상반신 박스 추정 (노트북 웹캠은 상반신만 보이는 경우가 많음)
        out = []
        small = cv2.resize(frame, (w // 2, h // 2))
        rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        for (x, y, bw, bh), wt in zip(rects, np.ravel(weights) if len(rects) else []):
            x, y, bw, bh = [v * 2 for v in (x, y, bw, bh)]
            out.append((x, y, x + bw, y + bh, float(min(0.99, 0.5 + 0.1 * float(wt)))))
        if not out:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x, y, fw, fh) in self.face.detectMultiScale(gray, 1.15, 5, minSize=(60, 60)):
                out.append((max(0, x - fw), y, min(w, x + 2 * fw), min(h, y + 4 * fh), 0.80))
        return out


def put(img, text, org, scale=0.55, color=(255, 255, 255), th=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def draw_panel(ctl, relay, mode_txt, stats, size=(300, 480)):
    """가상 선풍기: 위에서 본 Pan 방향 + 옆에서 본 Tilt 각도 + 지표"""
    W, H = size; p = np.full((H, W, 3), (40, 32, 26), np.uint8)
    put(p, 'VIRTUAL FAN (V3)', (14, 26), 0.6, (200, 230, 240))
    # Pan (top view)
    cx, cy, R = W // 2, 150, 95
    cv2.circle(p, (cx, cy), R, (90, 80, 70), 1, cv2.LINE_AA)
    for a in (-170, -90, 0, 90, 170):
        r = math.radians(a); cv2.line(p, (int(cx + (R - 8) * math.sin(-r)), int(cy - (R - 8) * math.cos(r))),
                                      (int(cx + R * math.sin(-r)), int(cy - R * math.cos(r))), (120, 110, 100), 1)
    cv2.line(p, (cx - R, cy + R + 10), (cx + R, cy + R + 10), (150, 150, 150), 3)   # 벽
    r = math.radians(ctl.pan); tip = (int(cx - R * 0.9 * math.sin(r)), int(cy - R * 0.9 * math.cos(r)))
    col = (219, 111, 47) if relay else (130, 130, 130)
    cv2.arrowedLine(p, (cx, cy), tip, col, 4, cv2.LINE_AA, tipLength=0.18)
    if relay:
        for k in (-1, 1):
            rr = r + k * math.radians(18)
            cv2.line(p, (cx, cy), (int(cx - R * math.sin(rr)), int(cy - R * math.cos(rr))), (160, 90, 40), 1, cv2.LINE_AA)
    put(p, f'PAN  {ctl.pan:+6.1f} deg', (14, 280), 0.55)
    # Tilt (side view)
    ox, oy, L = 60, 350, 110
    t = math.radians(ctl.tilt)
    cv2.line(p, (ox, oy), (int(ox + L * math.cos(t)), int(oy - L * math.sin(t))), col, 4, cv2.LINE_AA)
    for a in (-30, 45):
        ra = math.radians(a); cv2.line(p, (ox, oy), (int(ox + 40 * math.cos(ra)), int(oy - 40 * math.sin(ra))), (110, 100, 90), 1)
    put(p, f'TILT {ctl.tilt:+6.1f} deg', (150, 350), 0.5)
    put(p, f'STEP {ctl.steps:+d}', (150, 375), 0.5)
    put(p, f'MODE  {mode_txt}', (14, 410), 0.55, (120, 220, 255))
    put(p, f'FAN   {"ON" if relay else "OFF (relay)"}', (14, 435), 0.55, (120, 230, 120) if relay else (120, 120, 255))
    put(p, f'DET {stats[0]:5.1f}%  OFF {stats[1]:4.1f}%  FPS {stats[2]:4.1f}', (14, 462), 0.45, (200, 200, 200))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=None, help='웹캠 대신 영상 파일')
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--detector', default='auto', choices=['auto', 'yolo', 'hog'])
    ap.add_argument('--model', default='yolov8n.pt', help='YOLO 가중치 (기본 YOLOv8n, 처음 실행 시 자동 다운로드)')
    ap.add_argument('--ncnn', action='store_true', help='라즈베리파이 속도 향상: NCNN 형식으로 변환해 사용')
    ap.add_argument('--headless', action='store_true', help='화면 없이 실행(테스트용)')
    ap.add_argument('--frames', type=int, default=0, help='지정 프레임 후 종료(테스트용)')
    ap.add_argument('--thermal', default='auto', choices=['auto', 'none', 'mlx90640'], help='열화상 센서 (라즈베리파이)')
    ap.add_argument('--camera-moves', action='store_true', help='카메라가 선풍기 헤드와 함께 도는 실제 장치 (라즈베리파이)')
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video if a.video else a.cam)
    if not cap.isOpened():
        raise SystemExit('카메라/영상을 열 수 없습니다. macOS는 [시스템 설정 > 개인정보 보호 > 카메라]에서 터미널 권한을 허용하세요.')
    det = Detector(a.detector, a.model, a.ncnn); ctl = PanTiltController(CFG)
    judge = LivenessJudge(ThermalSensor(a.thermal)); use_liveness = True
    vfps = cap.get(cv2.CAP_PROP_FPS) if a.video else 0
    vfps = vfps if vfps and vfps > 1 else 15.0
    tan_h, tan_v = math.tan(math.radians(HFOV / 2)), math.tan(math.radians(VFOV / 2))
    mode, power, manual = 'auto', True, [0, 0]
    last_det, search_dir, t0, prev = -99.0, 1, time.time(), time.time()
    n_try = n_ok = 0; t_off = 0.0; fps = 0.0
    fname = datetime.datetime.now().strftime('tracking_log_%Y%m%d_%H%M%S.csv')
    fcsv = open(fname, 'w', newline='', encoding='utf-8-sig'); wr = csv.writer(fcsv)
    wr.writerow(['t_s', 'detected', 'conf', 'err_pan_deg', 'err_tilt_deg', 'pan_deg', 'tilt_deg', 'target_pan_deg', 'relay_on', 'mode',
                 'n_person_box', 'n_live', 'n_mannequin', 'target_id', 'target_state', 'target_reason'])
    print(f'[{det.label}] 실행 중 — 로그: {fname}  (Q: 종료)')
    k_frame = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1) if a.video is None else frame     # 거울 모드 (사용자가 보기 편하게)
        now = time.time(); dt = max(1e-3, min(0.2, now - prev)); prev = now; T = now - t0
        if a.video: dt = 1.0 / vfps; T = k_frame / vfps                   # 영상 파일은 영상 시간 기준
        fps = 0.9 * fps + 0.1 / dt
        h, w = frame.shape[:2]
        dets = det(frame); n_try += 1
        # ── 사람 / 마네킹 구분 ──
        bearing_of = lambda b: (ctl.pan if a.camera_moves else 0.0) - math.degrees(math.atan((((b[0] + b[2]) / 2) / w * 2 - 1) * tan_h))
        tracks, target = judge.update(frame, dets, T, dt, bearing_of, ctl.pan_v if a.camera_moves else 0.0, HFOV, VFOV)
        if not use_liveness:                                             # 비교용: 구분 없이 가장 큰 박스
            target = max(tracks, key=lambda tr: (tr.box[2] - tr.box[0]) * (tr.box[3] - tr.box[1])) if tracks else None
        box = (tuple(target.box) + (target.conf,)) if target else None
        ep = et = None
        if box:
            n_ok += 1; last_det = T
            x0, y0, x1, y1, conf = box
            cx = ((x0 + x1) / 2) / w * 2 - 1                            # -1(왼쪽) ~ 1(오른쪽)
            aim_y = 1 - (y0 + (y1 - y0) * 0.26) / h * 2                 # 가슴 높이 조준점
            ep = -math.degrees(math.atan(cx * tan_h)); et = math.degrees(math.atan(aim_y * tan_v))
            if mode == 'auto':
                ctl.tgt_pan = float(np.clip(ctl.pan + ep, CFG['pan_min'], CFG['pan_max']))
                ctl.tgt_tilt = ctl.tilt + et
            ax, ay = int((cx + 1) / 2 * w), int((1 - aim_y) / 2 * h)
            cv2.line(frame, (w // 2, h // 2), (ax, ay), (90, 200, 255), 1, cv2.LINE_AA); cv2.circle(frame, (ax, ay), 5, (90, 200, 255), -1)
        lost = T - last_det
        vmax = None
        if mode == 'manual':
            ctl.tgt_pan = float(np.clip(ctl.pan + manual[0] * 20, CFG['pan_min'], CFG['pan_max']))
            ctl.tgt_tilt = float(np.clip(ctl.tilt + manual[1] * 20, CFG['tilt_min'], CFG['tilt_max']))
            if manual == [0, 0]: ctl.tgt_pan = ctl.pan
            vmax = 40; manual = [0, 0]
        elif lost > CFG['search_after']:
            if abs(ctl.pan - search_dir * 80) < 4: search_dir *= -1
            ctl.tgt_pan, ctl.tgt_tilt = search_dir * 80.0, -8.0
            vmax = 15 if lost > CFG['off_delay'] else 30
        ctl.step(dt, vmax)
        relay = power and (mode == 'manual' or lost < CFG['off_delay'])
        if not relay: t_off += dt
        # 화면 구성
        draw_tracks(frame, tracks, target, put)
        n_live = sum(tr.state == 'LIVE' for tr in tracks); n_man = sum(tr.state == 'MANNEQUIN' for tr in tracks)
        put(frame, f'LIVENESS {"ON" if use_liveness else "OFF"} | person {n_live}  mannequin {n_man}  checking {len(tracks) - n_live - n_man}'
                   f' | thermal {"MLX90640" if judge.thermal and judge.thermal.ok else "none (motion only)"}', (10, 48), 0.5, (230, 230, 230))
        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
        if mode == 'manual': mode_txt = 'MANUAL (IR)'
        elif lost > CFG['off_delay']: mode_txt = 'IDLE (slow scan)'
        elif lost > CFG['search_after']: mode_txt = 'SEARCH'
        else: mode_txt = 'AUTO TRACK'
        put(frame, f'{det.label}  {mode_txt}  lost {min(lost, 99):.1f}s', (10, 24), 0.6)
        if ep is not None: put(frame, f'e_pan {ep:+.1f}  e_tilt {et:+.1f} deg', (10, h - 14), 0.55, (90, 200, 255))
        stats = (n_ok / max(1, n_try) * 100, t_off / max(1e-3, T) * 100, fps)
        panel = draw_panel(ctl, relay, mode_txt, stats, (300, h if h >= 480 else 480))
        if panel.shape[0] != h: frame = cv2.resize(frame, (int(w * panel.shape[0] / h), panel.shape[0]))
        out = np.hstack([frame, panel])
        if a.headless and k_frame == 40: cv2.imwrite('headless_preview.png', out)
        wr.writerow([f'{T:.3f}', int(box is not None), f'{box[4]:.2f}' if box else '', f'{ep:.2f}' if ep is not None else '',
                     f'{et:.2f}' if et is not None else '', f'{ctl.pan:.2f}', f'{ctl.tilt:.2f}', f'{ctl.tgt_pan:.2f}', int(relay), mode,
                     len(tracks), n_live, n_man, target.id if target else '', target.state if target else '', target.reason if target else ''])
        k_frame += 1
        if a.frames and k_frame >= a.frames: break
        if a.headless: continue
        cv2.imshow('V3 Auto-Tracking Fan - Webcam Simulation', out)
        k = cv2.waitKeyEx(1)
        if k in (ord('q'), ord('Q'), 27): break
        elif k in (ord('a'), ord('A')): mode = 'auto'; last_det = T
        elif k in (ord('m'), ord('M')): mode = 'manual'
        elif k in (ord('p'), ord('P')): power = not power
        elif k in (ord('l'), ord('L')): use_liveness = not use_liveness
        elif k in (ord('r'), ord('R')): n_try = n_ok = 0; t_off = 0.0; t0 = time.time() - 1e-3; last_det = -99
        elif k in (2424832, 63234, 81): mode = 'manual'; manual = [1, 0]     # ←
        elif k in (2555904, 63235, 83): mode = 'manual'; manual = [-1, 0]    # →
        elif k in (2490368, 63232, 82): mode = 'manual'; manual = [0, 1]     # ↑
        elif k in (2621440, 63233, 84): mode = 'manual'; manual = [0, -1]    # ↓
    fcsv.close(); cap.release()
    if not a.headless: cv2.destroyAllWindows()
    print(f'종료 — 검출률 {n_ok / max(1, n_try) * 100:.1f}%, 로그 저장: {fname}')


if __name__ == '__main__':
    main()
