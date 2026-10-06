# -*- coding: utf-8 -*-
"""
사람 / 마네킹(사진·포스터 포함) 구분 모듈 — V3 자동 추적 선풍기
Raspberry Pi 본체 코드와 웹캠 데모가 같이 쓰는 모듈입니다.

YOLO는 마네킹·사진 속 사람도 'person'으로 검출합니다. 그래서 검출된 객체마다
'살아 있는 사람인지'를 두 가지 근거로 따로 판정합니다.

  1) 열화상 (MLX90640, 선택 부품)  : 상반신 온도가 주변보다 +3 °C 이상이고 28 °C 이상 → 사람
                                     주변 온도와 비슷(+1.5 °C 미만) → 마네킹
  2) 미세 움직임 (소프트웨어, 무료) : 박스 안의 프레임 차이 − 박스 주변 배경의 프레임 차이
                                     (선풍기가 돌면서 생기는 화면 전체 흔들림은 빼고 '그 객체만의' 움직임을 봄)
                                     10초 안에 움직임 6회 이상 → 사람 / 15초 동안 한 번도 없음 → 마네킹(정지 객체)

판정 상태: UNKNOWN(판별 중) → LIVE(사람) / MANNEQUIN(마네킹·사진)
  · LIVE는 추적이 끊길 때까지 유지 → 잠든 사람처럼 가만히 있어도 선풍기가 꺼지지 않음
  · MANNEQUIN으로 판정된 위치(방위각)는 기억 → 다시 검출돼도 바로 무시 (움직이거나 뜨거우면 LIVE로 복귀)
  · 추적 대상은 LIVE 우선, 없으면 UNKNOWN. MANNEQUIN만 있으면 '사람 없음'으로 보고 탐색 → 자동 OFF
"""
import math, time, threading
from collections import deque
import numpy as np
import cv2

P = dict(
    iou_match=0.3, max_miss=15,                 # 추적기
    move_thr=0.8, live_events=6, live_window=10.0,  # 움직임 → LIVE
    static_sec=15.0, static_max_events=1,       # 움직임 없음 → MANNEQUIN (열화상 없을 때)
    hot_delta=3.0, hot_min=28.0, cold_delta=1.5,  # 열화상 판정 (°C)
    hot_reads=3, cold_reads=12,                 # 연속 판정 횟수
    revive_events=8, revive_window=5.0, revive_hot=5,  # MANNEQUIN → LIVE 복귀 조건
    memory_deg=5.0, memory_sec=600.0,           # 마네킹 위치 기억 (방위각 허용 오차, 유지 시간)
    cam_still_dps=8.0,                          # 이 속도(°/s) 이하일 때만 '정지 판정' 누적
)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    _next = 1

    def __init__(self, box, conf, t):
        self.id = Track._next; Track._next += 1
        self.box, self.conf = box, conf
        self.first_t = self.last_t = t
        self.misses = 0
        self.state = 'UNKNOWN'; self.reason = '판별 중'
        self.events = deque()                   # 움직임 이벤트 시각
        self.n_events = 0
        self.still_time = 0.0                   # 카메라가 멈춰 있던 누적 시간 (정지 판정용)
        self.hot = self.cold = 0
        self.temp = None; self.motion = 0.0
        self.bearing = None
        self.remembered = False


class IoUTracker:
    def __init__(self):
        self.tracks = []

    def update(self, dets, t):
        """dets: [(x0,y0,x1,y1,conf), ...] → 이번 프레임에 보인 트랙 목록"""
        unmatched = list(range(len(dets)))
        pairs = sorted(((iou(tr.box, dets[j][:4]), i, j) for i, tr in enumerate(self.tracks) for j in range(len(dets))), reverse=True)
        used_t, used_d = set(), set()
        for s, i, j in pairs:
            if s < P['iou_match'] or i in used_t or j in used_d: continue
            tr = self.tracks[i]; tr.box, tr.conf = tuple(dets[j][:4]), dets[j][4]; tr.last_t = t; tr.misses = 0
            used_t.add(i); used_d.add(j)
        for i, tr in enumerate(self.tracks):
            if i not in used_t: tr.misses += 1
        for j in range(len(dets)):
            if j not in used_d: self.tracks.append(Track(tuple(dets[j][:4]), dets[j][4], t))
        self.tracks = [tr for tr in self.tracks if tr.misses <= P['max_miss']]
        return [tr for tr in self.tracks if tr.misses == 0]


def relative_motion(prev_gray, gray, box, scale):
    """박스 안 움직임 − 박스 주변 배경 움직임 (0~255 밝기 단위). 카메라 자체 움직임을 상쇄."""
    h, w = gray.shape
    x0, y0, x1, y1 = [v * scale for v in box]
    bw, bh = x1 - x0, y1 - y0
    ix0, iy0, ix1, iy1 = int(x0 + 0.1 * bw), int(y0 + 0.05 * bh), int(x1 - 0.1 * bw), int(y1 - 0.1 * bh)
    ox0, oy0, ox1, oy1 = int(max(0, x0 - 0.3 * bw)), int(max(0, y0 - 0.2 * bh)), int(min(w, x1 + 0.3 * bw)), int(min(h, y1 + 0.2 * bh))
    if ix1 - ix0 < 4 or iy1 - iy0 < 4: return 0.0
    diff = cv2.absdiff(gray, prev_gray).astype(np.float32)
    inner = diff[iy0:iy1, ix0:ix1].mean()
    outer = diff[oy0:oy1, ox0:ox1]
    ring_sum = outer.sum() - diff[int(y0):int(y1), int(x0):int(x1)].sum()
    ring_n = outer.size - max(1, int(y1) - int(y0)) * max(1, int(x1) - int(x0))
    ring = ring_sum / ring_n if ring_n > 50 else float(np.median(diff))
    return float(inner - ring)


class ThermalSensor:
    """MLX90640 (32×24, 55°×35° 권장). Raspberry Pi에서 adafruit-circuitpython-mlx90640 설치 시 자동 사용.
    카메라 옆에 나란히 달아 광축을 맞춘다 (1 m 이상 거리에서는 시차 무시 가능)."""
    FOV_H, FOV_V = 55.0, 35.0

    def __init__(self, mode='auto'):
        self.frame = None; self.ok = False; self.sim = None
        if mode == 'none': return
        try:
            import board, busio, adafruit_mlx90640
            i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
            self.dev = adafruit_mlx90640.MLX90640(i2c)
            self.dev.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
            self.ok = True
            threading.Thread(target=self._loop, daemon=True).start()
        except Exception as ex:
            if mode == 'mlx90640': raise
            print('[안내] 열화상 센서 없음 → 미세 움직임만으로 판정합니다:', ex)

    def _loop(self):
        buf = [0.0] * 768
        while True:
            try:
                self.dev.getFrame(buf); self.frame = np.array(buf, dtype=np.float32).reshape(24, 32)
            except Exception:
                time.sleep(0.05)

    def region_temp(self, box, img_w, img_h, hfov, vfov):
        """상반신(박스 위쪽 45%)의 90백분위 온도, 주변(프레임 중앙값). 열화상 시야 밖이면 None."""
        f = self.frame
        if f is None: return None
        def to_cell(px, py):
            ax = math.atan((px / img_w * 2 - 1) * math.tan(math.radians(hfov / 2)))
            ay = math.atan((py / img_h * 2 - 1) * math.tan(math.radians(vfov / 2)))
            u = (math.tan(ax) / math.tan(math.radians(self.FOV_H / 2)) + 1) / 2 * 32
            v = (math.tan(ay) / math.tan(math.radians(self.FOV_V / 2)) + 1) / 2 * 24
            return u, v
        x0, y0, x1, y1 = box
        u0, v0 = to_cell(x0, y0); u1, v1 = to_cell(x1, y0 + 0.45 * (y1 - y0))
        u0, u1 = int(max(0, math.floor(u0))), int(min(32, math.ceil(u1)))
        v0, v1 = int(max(0, math.floor(v0))), int(min(24, math.ceil(v1)))
        if u1 - u0 < 1 or v1 - v0 < 1: return None
        reg = f[v0:v1, u0:u1]
        return float(np.percentile(reg, 90)), float(np.median(f))


class LivenessJudge:
    def __init__(self, thermal=None):
        self.tracker = IoUTracker()
        self.thermal = thermal
        self.memory = []                        # [(bearing_deg, t)]
        self.prev_gray = None; self.scale = 1.0
        self.target_id = None

    def _remembered(self, bearing, t):
        self.memory = [(b, tm) for b, tm in self.memory if t - tm < P['memory_sec']]
        return any(abs(b - bearing) < P['memory_deg'] for b, _ in self.memory)

    def update(self, frame, dets, t, dt, bearing_of, cam_speed_dps, hfov, vfov):
        """frame: BGR, dets: 사람 박스 목록, bearing_of(box)→방위각(°), cam_speed_dps: 카메라(Pan) 회전 속도"""
        h, w = frame.shape[:2]
        self.scale = 320.0 / w
        gray = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (320, int(h * self.scale))), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        seen = self.tracker.update(dets, t)
        cam_still = abs(cam_speed_dps) < P['cam_still_dps']
        for tr in seen:
            tr.bearing = bearing_of(tr.box)
            if tr.state == 'UNKNOWN' and not tr.remembered and self._remembered(tr.bearing, t):
                tr.state, tr.reason, tr.remembered = 'MANNEQUIN', '기억된 마네킹 위치', True
            # ① 미세 움직임
            if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
                tr.motion = relative_motion(self.prev_gray, gray, tr.box, self.scale)
                if tr.motion > P['move_thr']:
                    tr.events.append(t); tr.n_events += 1
            while tr.events and t - tr.events[0] > P['live_window']: tr.events.popleft()
            if cam_still: tr.still_time += dt
            # ② 열화상
            tr.temp = None
            if self.thermal is not None and self.thermal.frame is not None:
                r = self.thermal.region_temp(tr.box, w, h, hfov, vfov)
                if r:
                    t90, amb = r; tr.temp = (t90, amb)
                    if t90 >= max(P['hot_min'], amb + P['hot_delta']): tr.hot += 1; tr.cold = 0
                    elif t90 < amb + P['cold_delta']: tr.cold += 1; tr.hot = 0
            # ③ 상태 결정
            recent = len(tr.events)
            recent_short = sum(1 for e in tr.events if t - e < P['revive_window'])
            if tr.state == 'UNKNOWN':
                if tr.hot >= P['hot_reads']: tr.state, tr.reason = 'LIVE', '체온 감지'
                elif recent >= P['live_events']: tr.state, tr.reason = 'LIVE', '움직임 감지'
                elif tr.cold >= P['cold_reads']: tr.state, tr.reason = 'MANNEQUIN', '체온 없음'
                elif self.thermal_absent() and tr.still_time >= P['static_sec'] and tr.n_events <= P['static_max_events']:
                    tr.state, tr.reason = 'MANNEQUIN', f"{P['static_sec']:.0f}초간 움직임 없음"
            elif tr.state == 'MANNEQUIN':
                if tr.hot >= P['revive_hot'] or recent_short >= P['revive_events']:
                    tr.state, tr.reason = 'LIVE', '다시 움직임/체온'
                    self.memory = [(b, tm) for b, tm in self.memory if abs(b - tr.bearing) >= P['memory_deg']]
            if tr.state == 'MANNEQUIN' and not any(abs(b - tr.bearing) < P['memory_deg'] for b, _ in self.memory):
                self.memory.append((tr.bearing, t))
        self.prev_gray = gray
        # ④ 추적 대상 선택: LIVE > UNKNOWN, 이전 대상 유지(흔들림 방지), 그다음 가장 큰(가까운) 사람
        cand = [tr for tr in seen if tr.state in ('LIVE', 'UNKNOWN')]
        target = None
        if cand:
            best_rank = max(cand, key=lambda tr: (tr.state == 'LIVE', tr.id == self.target_id, (tr.box[2] - tr.box[0]) * (tr.box[3] - tr.box[1])))
            target = best_rank
        self.target_id = target.id if target else None
        return seen, target

    def thermal_absent(self):
        return self.thermal is None or self.thermal.frame is None


COLORS = {'LIVE': (80, 220, 120), 'UNKNOWN': (40, 200, 240), 'MANNEQUIN': (90, 90, 235)}
LABEL = {'LIVE': 'PERSON', 'UNKNOWN': 'CHECKING', 'MANNEQUIN': 'MANNEQUIN'}


def draw_tracks(img, tracks, target, put):
    for tr in tracks:
        x0, y0, x1, y1 = [int(v) for v in tr.box]
        c = COLORS[tr.state]; th = 3 if target is tr else 1
        cv2.rectangle(img, (x0, y0), (x1, y1), c, th)
        extra = f' {tr.temp[0]:.1f}C' if tr.temp else ''
        put(img, f'#{tr.id} {LABEL[tr.state]}{extra} m{tr.motion:+.1f}', (x0, max(18, y0 - 6)), 0.5, c)
        if tr.state == 'MANNEQUIN':
            cv2.line(img, (x0, y0), (x1, y1), c, 1); cv2.line(img, (x0, y1), (x1, y0), c, 1)
        if target is tr: put(img, 'TARGET', (x0, min(img.shape[0] - 6, y1 + 18)), 0.5, c)
