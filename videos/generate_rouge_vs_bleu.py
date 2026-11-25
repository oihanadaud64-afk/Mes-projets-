"""Rouge vs Bleu : qui va s'echapper ? -- generative TikTok video (1080x1920)."""
import math, random, subprocess, sys, wave
import numpy as np
import cairo

W, H, FPS = 1080, 1920, 30
CX, CY = W / 2, 1010
SUB = 10
DT = 1.0 / FPS / SUB
GRAV = 1100.0
BALL_R = 20
N_RINGS = 24
R_MIN, R_MAX = 95, 485
GAP = math.radians(44)
RING_W = 7
END_HOLD = 4.0
SR = 44100


def hsv(h, s=0.85, v=1.0):
    i = int(h * 6) % 6; f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


class Sim:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        r = self.rng
        self.rings = []
        for i in range(N_RINGS):
            rad = R_MIN + (R_MAX - R_MIN) * i / (N_RINGS - 1)
            spd = (0.7 + 0.9 * r.random()) * (1 if i % 2 == 0 else -1)
            self.rings.append({"r": rad, "a": r.random() * 2 * math.pi, "w": spd,
                               "alive": True, "col": hsv(i / N_RINGS * 0.85)})
        a = r.random() * 2 * math.pi
        sp = 650
        self.balls = [
            {"p": np.array([-30.0, -20.0]), "v": np.array([math.cos(a) * sp, math.sin(a) * sp]),
             "col": (1.0, 0.18, 0.25), "name": "ROUGE", "score": 0, "trail": []},
            {"p": np.array([30.0, -20.0]), "v": np.array([-math.cos(a) * sp, math.sin(a + 1) * sp]),
             "col": (0.15, 0.55, 1.0), "name": "BLEU", "score": 0, "trail": []},
        ]
        self.t = 0.0
        self.events = []  # (t, kind, data)
        self.lead_changes = 0
        self.leader = None
        self.done_t = None

    def inner(self):
        for k, ring in enumerate(self.rings):
            if ring["alive"]:
                return k
        return None

    def step(self):
        for ring in self.rings:
            ring["a"] += ring["w"] * DT
        for b in self.balls:
            b["v"][1] += GRAV * DT
            b["p"] += b["v"] * DT
        # ball-ball
        b1, b2 = self.balls
        d = b2["p"] - b1["p"]; dist = np.linalg.norm(d)
        if 0 < dist < 2 * BALL_R:
            n = d / dist
            rel = np.dot(b1["v"] - b2["v"], n)
            if rel > 0:
                b1["v"] -= rel * n; b2["v"] += rel * n
                self.events.append((self.t, "clack", None))
            ov = 2 * BALL_R - dist
            b1["p"] -= n * ov / 2; b2["p"] += n * ov / 2
        k = self.inner()
        if k is None:
            return
        ring = self.rings[k]
        R = ring["r"]
        for idx, b in enumerate(self.balls):
            dist = np.linalg.norm(b["p"])
            if dist > R:  # escaped through gap
                ring["alive"] = False
                b["score"] += 1
                self.events.append((self.t, "break", (k, idx)))
                lead = None
                if self.balls[0]["score"] != self.balls[1]["score"]:
                    lead = 0 if self.balls[0]["score"] > self.balls[1]["score"] else 1
                if lead is not None and self.leader is not None and lead != self.leader:
                    self.lead_changes += 1
                if lead is not None:
                    self.leader = lead
                if self.inner() is None:
                    self.done_t = self.t
                return
            if dist + BALL_R >= R:
                ang = math.atan2(b["p"][1], b["p"][0])
                delta = (ang - ring["a"] + math.pi) % (2 * math.pi) - math.pi
                half = GAP / 2 - math.asin(min(1, BALL_R / R))
                if abs(delta) < half:
                    continue
                n = b["p"] / dist
                vn = np.dot(b["v"], n)
                if vn > 0:
                    b["v"] -= 2 * vn * n
                    # tiny tangential kick to avoid periodic orbits, keep energy lively
                    tng = np.array([-n[1], n[0]])
                    b["v"] += tng * self.rng.uniform(-40, 40)
                    sp = np.linalg.norm(b["v"])
                    if sp < 700:
                        b["v"] *= 700 / sp
                    if sp > 1500:
                        b["v"] *= 1500 / sp
                    self.events.append((self.t, "bounce", idx))
                b["p"] = n * (R - BALL_R)

    def frame(self):
        for _ in range(SUB):
            if self.done_t is None:
                self.step()
            else:  # balls fly freely after the end
                for b in self.balls:
                    b["v"][1] += GRAV * DT
                    b["p"] += b["v"] * DT
            self.t += DT
        for b in self.balls:
            b["trail"].append(b["p"].copy())
            b["trail"] = b["trail"][-14:]


def evaluate(seed, tmax=90):
    s = Sim(seed)
    while s.done_t is None and s.t < tmax:
        s.frame()
    return s


# ---------------- rendering ----------------
def text_c(ctx, txt, y, size, col, bold=True, outline=True):
    ctx.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    ext = ctx.text_extents(txt)
    x = W / 2 - ext.width / 2 - ext.x_bearing
    ctx.move_to(x, y)
    ctx.text_path(txt)
    if outline:
        ctx.set_source_rgb(0, 0, 0); ctx.set_line_width(size * 0.14)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND); ctx.stroke_preserve()
    ctx.set_source_rgb(*col); ctx.fill()


def render(seed, out_video):
    sim = Sim(seed)
    prng = random.Random(seed + 999)
    particles = []
    flashes = []  # (t0, radius, col)
    ev_seen = 0
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", out_video],
                          stdin=subprocess.PIPE)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    bg = cairo.LinearGradient(0, 0, 0, H)
    bg.add_color_stop_rgb(0, 0.04, 0.03, 0.10); bg.add_color_stop_rgb(1, 0.01, 0.01, 0.03)
    end_frames = None
    fi = 0
    while True:
        sim.frame()
        t = sim.t
        # new events -> particles
        while ev_seen < len(sim.events):
            et, kind, data = sim.events[ev_seen]; ev_seen += 1
            if kind == "break":
                k, idx = data
                ring = sim.rings[k]
                for _ in range(70):
                    a = prng.random() * 2 * math.pi
                    sp = prng.uniform(150, 700)
                    particles.append([CX + math.cos(a) * ring["r"], CY + math.sin(a) * ring["r"],
                                      math.cos(a) * sp, math.sin(a) * sp, 1.0, ring["col"]])
                flashes.append((t, ring["r"], sim.balls[idx]["col"]))
        # draw
        ctx.set_source(bg); ctx.paint()
        # rings
        for ring in sim.rings:
            if not ring["alive"]:
                continue
            a0 = ring["a"] + GAP / 2; a1 = ring["a"] - GAP / 2 + 2 * math.pi
            for lw, alpha in ((RING_W * 3.2, 0.18), (RING_W, 1.0)):
                ctx.set_source_rgba(*ring["col"], alpha)
                ctx.set_line_width(lw); ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.new_path(); ctx.arc(CX, CY, ring["r"], a0, a1); ctx.stroke()
        # flashes
        for f in flashes[:]:
            age = t - f[0]
            if age > 0.5:
                flashes.remove(f); continue
            ctx.set_source_rgba(*f[2], 0.6 * (1 - age / 0.5))
            ctx.set_line_width(14 * (1 - age / 0.5) + 2)
            ctx.new_path(); ctx.arc(CX, CY, f[1] + age * 300, 0, 2 * math.pi); ctx.stroke()
        # particles
        for p in particles[:]:
            p[0] += p[2] / FPS; p[1] += p[3] / FPS; p[3] += 600 / FPS; p[4] -= 1 / (FPS * 1.1)
            if p[4] <= 0:
                particles.remove(p); continue
            ctx.set_source_rgba(*p[5], p[4])
            ctx.new_path(); ctx.arc(p[0], p[1], 4, 0, 2 * math.pi); ctx.fill()
        # balls
        for b in sim.balls:
            tr = b["trail"]
            for i, q in enumerate(tr[:-1]):
                al = (i + 1) / len(tr) * 0.45
                ctx.set_source_rgba(*b["col"], al)
                ctx.new_path(); ctx.arc(CX + q[0], CY + q[1], BALL_R * (0.4 + 0.6 * (i + 1) / len(tr)), 0, 2 * math.pi); ctx.fill()
            x, y = CX + b["p"][0], CY + b["p"][1]
            g = cairo.RadialGradient(x, y, BALL_R * 0.5, x, y, BALL_R * 2.8)
            g.add_color_stop_rgba(0, *b["col"], 0.55); g.add_color_stop_rgba(1, *b["col"], 0)
            ctx.set_source(g); ctx.new_path(); ctx.arc(x, y, BALL_R * 2.8, 0, 2 * math.pi); ctx.fill()
            ctx.set_source_rgb(*b["col"]); ctx.new_path(); ctx.arc(x, y, BALL_R, 0, 2 * math.pi); ctx.fill()
            ctx.set_source_rgba(1, 1, 1, 0.9); ctx.set_line_width(3)
            ctx.new_path(); ctx.arc(x, y, BALL_R, 0, 2 * math.pi); ctx.stroke()
            ctx.set_source_rgba(1, 1, 1, 0.6); ctx.new_path(); ctx.arc(x - 6, y - 6, 5, 0, 2 * math.pi); ctx.fill()
        # HUD
        text_c(ctx, "ROUGE OU BLEU ?", 175, 70, (1, 1, 1))
        text_c(ctx, "QUI VA GAGNER ?", 260, 70, (1, 0.85, 0.2))
        r_, b_ = sim.balls
        # scoreboard
        ctx.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        for i, b in enumerate(sim.balls):
            cx = W / 2 + (-210 if i == 0 else 210)
            ctx.set_source_rgba(*b["col"], 0.25)
            ctx.new_path(); ctx.rectangle(cx - 170, 330, 340, 170); ctx.fill()
            ctx.set_source_rgb(*b["col"]); ctx.set_line_width(5)
            ctx.new_path(); ctx.rectangle(cx - 170, 330, 340, 170); ctx.stroke()
            for txt, yy, sz in ((b["name"], 395, 50), (str(b["score"]), 480, 80)):
                ctx.set_font_size(sz)
                e = ctx.text_extents(txt)
                ctx.move_to(cx - e.width / 2 - e.x_bearing, yy)
                ctx.set_source_rgb(1, 1, 1); ctx.show_text(txt)
        left = sum(1 for r in sim.rings if r["alive"])
        if sim.done_t is None:
            text_c(ctx, f"Anneaux restants : {left}", 1590, 46, (0.85, 0.85, 0.95), outline=False)
            text_c(ctx, "Commente ta couleur !", 1720, 62, (1, 1, 1))
        else:
            win = 0 if r_["score"] > b_["score"] else 1 if b_["score"] > r_["score"] else None
            age = t - sim.done_t
            sc = min(1, age / 0.3)
            if win is None:
                text_c(ctx, "ÉGALITÉ !", 1640, 120 * sc + 1, (1, 1, 1))
            else:
                text_c(ctx, f"{sim.balls[win]['name']} GAGNE !", 1640, 120 * sc + 1, sim.balls[win]["col"])
            text_c(ctx, "Abonne-toi pour la revanche", 1760, 50, (1, 1, 1))
            if end_frames is None:
                end_frames = int(END_HOLD * FPS)
        surf.flush()
        ff.stdin.write(bytes(surf.get_data()))
        fi += 1
        if end_frames is not None:
            end_frames -= 1
            if end_frames <= 0:
                break
    ff.stdin.close(); ff.wait()
    return sim, fi / FPS


# ---------------- audio ----------------
def midi_hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def tone(freq, dur, amp, decay):
    n = int(dur * SR); t = np.arange(n) / SR
    env = np.exp(-t / decay) * np.minimum(1, t / 0.004)
    s = (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t)
         + 0.12 * np.sin(6 * np.pi * freq * t))
    return amp * env * s


def audio(sim, total, out_wav):
    # public-domain melody: Ode to Joy (Beethoven), one octave up
    mel = [76, 76, 77, 79, 79, 77, 76, 74, 72, 72, 74, 76, 76, 74, 74,
           76, 76, 77, 79, 79, 77, 76, 74, 72, 72, 74, 76, 74, 72, 72,
           74, 74, 76, 72, 74, 76, 77, 76, 72, 74, 76, 77, 76, 74, 72, 74, 67,
           76, 76, 77, 79, 79, 77, 76, 74, 72, 72, 74, 76, 74, 72, 72]
    buf = np.zeros(int((total + 2) * SR))
    last = -1.0; mi = 0
    def add(t0, s):
        i = int(t0 * SR); j = min(len(buf), i + len(s))
        if i < len(buf): buf[i:j] += s[:j - i]
    for t0, kind, data in sim.events:
        if kind == "bounce":
            if t0 - last < 0.06:
                continue
            last = t0
            add(t0, tone(midi_hz(mel[mi % len(mel)]), 0.6, 0.28, 0.18)); mi += 1
        elif kind == "break":
            base = 84 + (data[0] % 5)
            for j, iv in enumerate((0, 4, 7, 12)):
                add(t0 + j * 0.035, tone(midi_hz(base + iv), 0.5, 0.12, 0.12))
            n = int(0.12 * SR)
            add(t0, np.random.default_rng(int(t0 * 1000)).standard_normal(n) * np.exp(-np.arange(n) / SR / 0.03) * 0.08)
    if sim.done_t is not None:
        for j, m in enumerate((72, 76, 79, 84)):
            add(sim.done_t + j * 0.12, tone(midi_hz(m), 1.8, 0.2, 0.6))
    buf = buf[:int(total * SR)]
    buf = np.tanh(buf * 1.2) * 0.85
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "search":
        best = []
        for seed in range(int(sys.argv[2]), int(sys.argv[3])):
            s = evaluate(seed)
            if s.done_t is None:
                continue
            d = abs(s.balls[0]["score"] - s.balls[1]["score"])
            print(seed, round(s.done_t, 1), s.balls[0]["score"], s.balls[1]["score"], s.lead_changes, flush=True)
    else:
        seed = int(sys.argv[2]); out = sys.argv[3]
        sim, total = render(seed, out + ".video.mp4")
        audio(sim, total, out + ".wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out + ".video.mp4", "-i", out + ".wav",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        "-movflags", "+faststart", out], check=True)
        print("done", total)
