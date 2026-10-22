import sys, json, os
import cv2, numpy as np
from PIL import Image
from rembg import new_session, remove

UP = '/root/.claude/uploads/b0b51825-4b6b-5d06-ad40-5df886bd9352/'
VID = UP + '1bc8e65c-VID-20261021-WA00132.mp4'
BG = UP + '915d0877-image.jpg'
OUT_W, OUT_H = 720, 1280
FPS = 30

# (start, end, extra rotation deg after 90 CCW, scale, x-shift, y-offset below bottom)
SEGMENTS = {
    'A': (2.6, 5.3, 5, 1.6, 0, 70),
    'B': (8.4, 12.9, 5, 1.6, 0, 70),
    'C': (18.4, 21.9, -28, 1.75, 0, 380),
    'D': (27.4, 29.2, 5, 1.6, 0, 70),
}

# horizontal keep-window (fraction of output width) to drop arm/towel at the sides
XWIN = {'A': (0.0, 0.76), 'B': (0.0, 0.86), 'C': (0.04, 1.0), 'D': (0.0, 1.0)}

GAIN = {'A': 1.0, 'B': 1.03, 'C': 1.0, 'D': 1.18}

s_body = new_session('u2net_human_seg')
s_hair = new_session('isnet-general-use')

bg0 = cv2.imread(BG).astype(np.float32) / 255.0
bg0 = cv2.resize(bg0, (OUT_W, OUT_H))


def grade_bg(bg):
    bg = cv2.GaussianBlur(bg, (0, 0), 2.2)  # light depth of field
    return np.clip(bg * np.array([0.96, 1.0, 1.04]), 0, 1)  # BGR: a touch warmer


BGG = grade_bg(bg0)
BG_SOFT = cv2.GaussianBlur(BGG, (0, 0), 25)


def bg_at(t_global, total):
    # slow push-in on the background so the scene feels alive
    z = 1.0 + 0.06 * (t_global / max(total, 1e-6))
    M = cv2.getRotationMatrix2D((OUT_W * 0.5, OUT_H * 0.45), 0, z)
    return (cv2.warpAffine(BGG, M, (OUT_W, OUT_H), borderMode=cv2.BORDER_REFLECT),
            cv2.warpAffine(BG_SOFT, M, (OUT_W, OUT_H), borderMode=cv2.BORDER_REFLECT))


def segment(rgb_u8):
    pil = Image.fromarray(rgb_u8)
    body = np.asarray(remove(pil, session=s_body, only_mask=True), np.float32) / 255
    hair = np.asarray(remove(pil, session=s_hair, only_mask=True), np.float32) / 255
    # keep only the main subject (drop towel / stray blobs)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((body > 0.4).astype(np.uint8))
    main = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else (body > 0.4)
    body = body * cv2.GaussianBlur(cv2.dilate(main.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 3)
    near = cv2.dilate(main.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(np.float32)
    near = cv2.GaussianBlur(near, (0, 0), 6)
    a = np.maximum(body, hair * near)
    return a


def refine(alpha, bgr):
    guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    a = cv2.ximgproc.guidedFilter(guide, (alpha * 255).astype(np.uint8), 4, 1e-2 * 255 * 255)
    a = a.astype(np.float32) / 255
    a = np.clip((a - 0.06) / 0.88, 0, 1)
    return a


def relight(img, alpha, lvl):
    """img BGR float. Remove haze/flare veil, add warm directional sun + rim."""
    black, white = lvl
    x = np.clip((img - black) / np.maximum(white - black, 1e-3), 0, 1)
    # gentle S-curve for contrast
    x = x + 0.22 * x * (1 - x) * (2 * x - 1)
    x = np.clip(0.5 + (x - 0.5) * 1.12, 0, 1)
    # saturation
    hsv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.18, 0, 255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255
    # warm golden balance to match the ipê light
    x = np.clip(x * np.array([0.90, 1.0, 1.08]), 0, 1)
    h, w = alpha.shape
    # directional key light from upper-right
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = (xx / w) * 0.6 + (1 - yy / h) * 0.8
    d = (d - d.min()) / (d.max() - d.min())
    key = 0.82 + 0.30 * d
    x = x * key[..., None]
    # warm soft light on the lit side
    warm = np.array([0.55, 0.85, 1.0], np.float32)
    x = x + 0.08 * (d ** 2)[..., None] * warm
    # rim light along the silhouette, stronger towards the sun
    er = cv2.erode(alpha, np.ones((9, 9), np.uint8))
    rim = cv2.GaussianBlur(np.clip(alpha - er, 0, 1), (0, 0), 3) * (0.3 + 0.7 * d)
    x = x + 0.35 * rim[..., None] * np.array([0.45, 0.85, 1.0])
    return np.clip(x, 0, 1)


def rotate_rgba(rgb, a, ang, scale):
    h, w = a.shape
    nw, nh = int(w * scale), int(h * scale)
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if ang:
        c = (nw / 2, nh / 2)
        M = cv2.getRotationMatrix2D(c, ang, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        bw, bh = int(nh * sin + nw * cos), int(nh * cos + nw * sin)
        M[0, 2] += bw / 2 - c[0]
        M[1, 2] += bh / 2 - c[1]
        rgb = cv2.warpAffine(rgb, M, (bw, bh), flags=cv2.INTER_CUBIC, borderValue=0)
        a = cv2.warpAffine(a, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0)
    return np.clip(rgb, 0, 1), np.clip(a, 0, 1)


def composite(person, a, bg, bg_soft, xshift, ybelow, cx, xwin=(0, 1)):
    H, W = a.shape
    ox = int(OUT_W / 2 - cx) + xshift
    oy = OUT_H + ybelow - H
    canvas_p = np.zeros((OUT_H, OUT_W, 3), np.float32)
    canvas_a = np.zeros((OUT_H, OUT_W), np.float32)
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + W, OUT_W), min(oy + H, OUT_H)
    canvas_p[y0:y1, x0:x1] = person[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    canvas_a[y0:y1, x0:x1] = a[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    fx = np.arange(OUT_W) / OUT_W
    win = np.clip((fx - xwin[0]) / 0.06 + 1, 0, 1) * np.clip((xwin[1] - fx) / 0.06 + 1, 0, 1)
    canvas_a *= win[None, :]
    # light wrap: let background light spill over the edges
    inner = cv2.GaussianBlur(canvas_a, (0, 0), 10)
    wrap = np.clip((1 - inner) * canvas_a, 0, 1) * 0.55
    canvas_p = canvas_p * (1 - wrap[..., None]) + bg_soft * wrap[..., None]
    # soft contact shadow under the subject
    sh = cv2.GaussianBlur(canvas_a, (0, 0), 22)
    sh = np.roll(sh, 18, axis=0) * 0.25
    base = bg * (1 - sh[..., None])
    out = canvas_p * canvas_a[..., None] + base * (1 - canvas_a[..., None])
    # global warm sun glow from upper right + gentle vignette
    yy, xx = np.mgrid[0:OUT_H, 0:OUT_W].astype(np.float32)
    g = np.exp(-(((xx - OUT_W * 0.95) / 520) ** 2 + ((yy + 80) / 620) ** 2))
    out = out + 0.10 * g[..., None] * np.array([0.6, 0.9, 1.0])
    v = 1 - 0.18 * (((xx / OUT_W - 0.5) ** 2 + (yy / OUT_H - 0.5) ** 2) * 2)
    return np.clip(out * v[..., None], 0, 1)


def run(seg_key, out_path, t_offset, total, only_frames=None):
    st, en, ang, scale, xshift, ybelow = SEGMENTS[seg_key]
    cap = cv2.VideoCapture(VID)
    f0, f1 = int(round(st * FPS)), int(round(en * FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    writer = None
    if out_path:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (OUT_W, OUT_H))
    prev_a, lvl, cx = None, None, None
    outs = []
    for i in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        if only_frames is not None and (i - f0) not in only_frames:
            continue
        fr = cv2.rotate(fr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        a = segment(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        a = refine(a, fr)
        if prev_a is not None and only_frames is None:
            a = 0.6 * a + 0.4 * prev_a
        prev_a = a
        # feather the original frame borders so cut hair doesn't end in a hard line
        hh, ww = a.shape
        ry = np.clip(np.arange(hh) / 40.0, 0, 1)[:, None]
        rx = np.minimum(np.clip(np.arange(ww) / 40.0, 0, 1), np.clip(np.arange(ww)[::-1] / 40.0, 0, 1))[None, :]
        a = a * ry * rx
        img = fr.astype(np.float32) / 255
        m = a > 0.7
        px = img[m] if m.sum() > 500 else img.reshape(-1, 3)
        cur = (np.percentile(px, 1.5, axis=0), np.percentile(px, 99.6, axis=0))
        lvl = cur if lvl is None else tuple(0.85 * l + 0.15 * c for l, c in zip(lvl, cur))
        lit = relight(img, a, lvl)
        lit = np.clip(lit * GAIN[seg_key], 0, 1)
        p, pa = rotate_rgba(lit, a, ang, scale)
        # horizontal centre of the head (top 35% of the subject)
        ys, xs = np.nonzero(pa > 0.5)
        if len(xs):
            top = ys < np.percentile(ys, 35)
            c = float(np.median(xs[top]))
            cx = c if cx is None else 0.92 * cx + 0.08 * c
        t = t_offset + (i - f0) / FPS
        bg, bgs = bg_at(t, total)
        out = composite(p, pa, bg, bgs, xshift, ybelow, cx if cx is not None else p.shape[1] / 2, XWIN[seg_key])
        o8 = (out * 255).astype(np.uint8)
        if writer:
            writer.write(o8)
        outs.append(o8)
        print(seg_key, i - f0, flush=True)
    if writer:
        writer.release()
    return outs


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'test':
        tiles = []
        for k in SEGMENTS:
            tiles += run(k, None, 0, 1, only_frames={10})
        cv2.imwrite(sys.argv[2], cv2.resize(np.hstack(tiles), None, fx=0.5, fy=0.5))
    else:
        k = mode
        offs = {'A': 0, 'B': 2.7, 'C': 7.2, 'D': 10.7}
        run(k, sys.argv[2], offs[k], 12.5)
