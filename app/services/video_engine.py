from __future__ import annotations

import subprocess
import shutil
import textwrap
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from app.core.logger import logger
from app.models.schemas import Clip, SocialCoverStyle, TiktokCaptionStyle

PROFILE_MAP = {
    "SD (720p)": {"preset": "veryfast", "crf": "24", "height": 720},
    "HD (1080p)": {"preset": "medium", "crf": "21", "height": 1080},
    "2K (1440p)": {"preset": "slow", "crf": "20", "height": 1440},
    "4K (2160p)": {"preset": "slow", "crf": "18", "height": 2160},
}
FALLBACK_PROFILE = {"preset": "medium", "crf": "21", "height": 1080}

# Exportação: <pasta do projeto>/clip_01/clipe.mp4, capa.jpg, descricao_redes.txt
CLIP_SESSION_SUBDIR_FMT = "clip_{:02d}"
EXPORT_CLIP_VIDEO_FILENAME = "clipe.mp4"
EXPORT_CLIP_COVER_FILENAME = "capa.jpg"
EXPORT_CLIP_SOCIAL_FILENAME = "descricao_redes.txt"
CLIP_BASE_FILENAME = "clipe_base.mp4"
CLIP_MOVIEPY_TEMP = "_clipe_moviepy.mp4"


def clip_session_subdirectory(session_root: Path, clip_index: int) -> Path:
    """Subpasta dedicada a um clipe (índice base 1)."""
    return session_root / CLIP_SESSION_SUBDIR_FMT.format(clip_index)


def _ffmpeg_has_encoder(encoder: str) -> bool:
    """
    Detecta se o FFmpeg tem um encoder disponível (ex.: h264_nvenc).
    Mantém fallback seguro para libx264 se não houver.
    """
    if not shutil.which("ffmpeg"):
        return False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except Exception:
        return False
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return encoder in out


def get_export_dimensions(
    resolution: str,
    export_quality: str,
    aspect_ratio: str,
) -> tuple[int, int]:
    profile = PROFILE_MAP.get(export_quality) or PROFILE_MAP.get(resolution) or FALLBACK_PROFILE
    is_vertical = "9:16" in aspect_ratio
    base_h = int(profile["height"])
    base_w = int(round(base_h * (9 / 16 if is_vertical else 16 / 9)))
    target_w = base_w - (base_w % 2)
    target_h = base_h - (base_h % 2)
    return target_w, target_h


def tiktok_subtitle_style_sizes(target_w: int, target_h: int) -> tuple[int, int]:
    """Mesmos parâmetros de fonte/margem usados nas legendas TikTok (.ass)."""
    style_font_size = max(34, int(round(target_h * 0.06)))
    style_margin_v = max(70, int(round(target_h * 0.09)))
    return style_font_size, style_margin_v


def tiktok_ass_v4_style_block(
    target_w: int,
    target_h: int,
    style: Optional[TiktokCaptionStyle] = None,
) -> str:
    """Bloco [V4+ Styles] (Format + Style Default) dos clipes com legenda TikTok."""
    fs_base, mv_base = tiktok_subtitle_style_sizes(target_w, target_h)
    if style is None:
        return (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,Arial,{fs_base},&H00FFFFFF,&H0000E5FF,&H00101010,&H80000000,1,0,0,0,"
            f"100,100,0,0,1,4,1,2,80,80,{mv_base},1"
        )
    fs = style.font_size if style.font_size and style.font_size > 0 else fs_base
    mv = style.margin_v if style.margin_v and style.margin_v > 0 else mv_base
    b_flag = -1 if style.bold else 0
    i_flag = -1 if style.italic else 0
    ff = (style.font_family or "Arial").replace(",", " ")
    return (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{ff},{fs},{style.primary_color_ass},{style.secondary_color_ass},"
        f"{style.outline_color_ass},&H80000000,{b_flag},{i_flag},0,0,{style.scale_x},{style.scale_y},0,0,1,"
        f"{style.outline},{style.shadow},2,80,80,{mv},1"
    )


def tiktok_cover_ass_v4_style_block(
    target_w: int,
    target_h: int,
    style: Optional[SocialCoverStyle] = None,
) -> str:
    """
    Estilo da capa: padrão histórico (amarelo-ouro, negrito, contorno forte).
    Com ``style`` preenchido (personalização na UI), usa esses campos.
    """
    fs_base, mv_base = tiktok_subtitle_style_sizes(target_w, target_h)
    if style is None:
        return (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Cover,Arial,{fs_base},&H0000D7FF,&H0000E5FF,&H00000000,&H80000000,-1,0,0,0,"
            f"102,102,0,0,1,6,3,2,80,80,{mv_base},1"
        )
    fs = style.font_size if style.font_size and style.font_size > 0 else fs_base
    mv = style.margin_v if style.margin_v and style.margin_v > 0 else mv_base
    b_flag = -1 if style.bold else 0
    i_flag = -1 if style.italic else 0
    ff = (style.font_family or "Arial").replace(",", " ")
    return (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cover,{ff},{fs},{style.primary_color_ass},{style.secondary_color_ass},"
        f"{style.outline_color_ass},&H80000000,{b_flag},{i_flag},0,0,{style.scale_x},{style.scale_y},0,0,1,"
        f"{style.outline},{style.shadow},2,80,80,{mv},1"
    )


def _ass_escape_basic(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _ffmpeg_extract_frame_png_bytes(video_path: Path, t_sec: float) -> Optional[bytes]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, t_sec):.3f}",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-c:v",
        "png",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode(errors="replace")[:400]
        logger.debug("FFmpeg extrair frame em %.3fs falhou: %s", t_sec, err)
        return None
    return proc.stdout


def _iou_rects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = float(aw * ah + bw * bh) - inter
    return inter / union if union > 0 else 0.0


def _nms_face_rects(
    rects: List[Tuple[int, int, int, int]], iou_thresh: float = 0.35
) -> List[Tuple[int, int, int, int]]:
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept: List[Tuple[int, int, int, int]] = []
    for r in rects:
        if all(_iou_rects(r, k) < iou_thresh for k in kept):
            kept.append(r)
    return kept


def _haar_detect_rects(
    cascade: Any,
    gray: Any,
    *,
    scale_factor: float,
    min_neighbors: int,
    min_side: int,
) -> List[Tuple[int, int, int, int]]:
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_side, min_side),
        flags=0,
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def _face_centers_from_gray(gray: Any, fw: int, fh: int) -> List[Tuple[float, float, float]]:
    """
    Retorna lista de (cx, cy, score) em coordenadas normalizadas [0,1], score para desempate.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return []

    haarc = cv2.data.haarcascades
    cascade_names = (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_profileface.xml",
    )
    frame_area = float(max(1, fw * fh))
    min_px = max(18, int(round(min(fw, fh) * 0.04)))
    param_sets = (
        (1.05, 3),
        (1.08, 2),
        (1.12, 2),
    )

    all_rects: List[Tuple[int, int, int, int]] = []
    for name in cascade_names:
        path = haarc + name
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            continue
        for sf, mn in param_sets:
            all_rects.extend(
                _haar_detect_rects(cascade, gray, scale_factor=sf, min_neighbors=mn, min_side=min_px)
            )

    merged = _nms_face_rects(all_rects, iou_thresh=0.32)
    out: List[Tuple[float, float, float]] = []
    for (x, y, w, h) in merged:
        area = float(w * h)
        if area < frame_area * 0.00035:
            continue
        cx = (x + w / 2.0) / float(fw)
        cy = (y + h / 2.0) / float(fh)
        area_n = area / frame_area
        # Preferir rostos maiores e um pouco mais centrados (reduz falsos positivos nas bordas).
        edge_pen = 1.0 - 0.22 * (abs(cx - 0.5) + abs(cy - 0.5))
        score = area_n * max(0.35, edge_pen)
        out.append((cx, cy, score))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def _best_face_center_ratios(frame: Any) -> Optional[Tuple[float, float]]:
    """
    Melhor estimativa do centro do rosto principal (cx, cy) em [0,1].
    Usa várias cascatas, escalas Haar, NMS e tentativas em pirâmide (rosto pequeno / frame grande).
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Deteção em resolução moderada: coords normalizadas são invariantes; 4K fica muito mais leve.
    gh, gw = gray.shape[:2]
    max_side = max(gw, gh)
    det_max = 1024
    if max_side > det_max:
        ds = det_max / float(max_side)
        work = cv2.resize(gray, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    else:
        work = gray
    fh, fw = work.shape[:2]
    candidates: List[Tuple[float, float, float]] = _face_centers_from_gray(work, fw, fh)

    # Rostos muito pequenos no plano de deteção: ampliar levemente.
    if not candidates:
        big = cv2.resize(work, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
        bh, bw = big.shape[:2]
        for cx, cy, sc in _face_centers_from_gray(big, bw, bh):
            candidates.append((cx, cy, sc * 0.88))

    if not candidates:
        return None
    best_cx, best_cy, _ = max(candidates, key=lambda t: t[2])
    return (max(0.0, min(1.0, best_cx)), max(0.0, min(1.0, best_cy)))


def detect_face_center_ratios_at_time(video_path: Path, t_sec: float) -> Optional[Tuple[float, float]]:
    raw = _ffmpeg_extract_frame_png_bytes(video_path, t_sec)
    if not raw:
        return None
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return _best_face_center_ratios(frame)


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.5
    s = sorted(xs)
    m = len(s) // 2
    return float(s[m]) if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _stratified_clip_sample_times(clip_start: float, clip_end: float, n_total: int) -> List[float]:
    """
    Distribui instantes ao longo do clipe (vários segmentos temporais).
    Assim, após um corte de câmara, ainda há amostras no plano anterior e no novo —
    o cluster dominante reflete o enquadramento onde o rosto aparece mais tempo.
    """
    duration = max(0.1, clip_end - clip_start)
    n_seg = 4
    per = max(1, n_total // n_seg)
    times: List[float] = []
    for seg in range(n_seg):
        t0 = clip_start + duration * seg / n_seg
        t1 = clip_start + duration * (seg + 1) / n_seg
        span = max(0.02, t1 - t0)
        for k in range(per):
            frac = (k + 0.5) / per
            times.append(t0 + span * frac)
    while len(times) < n_total:
        times.append(clip_start + duration * (len(times) + 0.5) / n_total)
    return times[:n_total]


def _dominant_spatial_cluster_median(
    points: List[Tuple[float, float]],
    grid_n: int = 7,
) -> Tuple[float, float]:
    """
    Agrupa posições (cx, cy) numa grelha 2D e escolhe a região com mais detecções
    (vizinhança 3×3), depois mediana dentro desse conjunto. Robustez a mudanças de câmara:
    um único plano errado não domina sobre a maioria dos frames no outro enquadramento.
    """
    if not points:
        return (0.5, 0.5)
    if len(points) == 1:
        return (max(0.0, min(1.0, points[0][0])), max(0.0, min(1.0, points[0][1])))

    buckets: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    for cx, cy in points:
        bx = min(grid_n - 1, max(0, int(cx * grid_n)))
        by = min(grid_n - 1, max(0, int(cy * grid_n)))
        buckets[(bx, by)].append((cx, cy))

    def cell_score(bx: int, by: int) -> float:
        sc = float(len(buckets.get((bx, by), [])))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sc += len(buckets.get((bx + dx, by + dy), [])) * 0.45
        return sc

    best_cell: Optional[Tuple[int, int]] = None
    best_sc = -1.0
    for (bx, by) in buckets:
        sc = cell_score(bx, by)
        if sc > best_sc:
            best_sc = sc
            best_cell = (bx, by)

    if best_cell is None:
        mx = _median([p[0] for p in points])
        my = _median([p[1] for p in points])
        return (max(0.0, min(1.0, mx)), max(0.0, min(1.0, my)))

    bx, by = best_cell
    pooled: List[Tuple[float, float]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            pooled.extend(buckets.get((bx + dx, by + dy), []))

    # Se o "vencedor" é muito pequeno face ao total, há vários planos equiparados — mediana global.
    min_support = max(2, int(round(0.2 * len(points))))
    if len(pooled) < min_support:
        mx = _median([p[0] for p in points])
        my = _median([p[1] for p in points])
        return (max(0.0, min(1.0, mx)), max(0.0, min(1.0, my)))

    mx = _median([p[0] for p in pooled])
    my = _median([p[1] for p in pooled])
    return (max(0.0, min(1.0, mx)), max(0.0, min(1.0, my)))


def _longest_temporal_run_median(
    ordered_hits: List[Tuple[float, float]],
    jump_thresh: float = 0.16,
) -> Optional[Tuple[float, float]]:
    """
    Secundário: maior sequência temporal de detecções com saltos pequenos (mesmo plano contínuo).
    """
    if len(ordered_hits) < 3:
        return None
    best_len = 0
    best_slice: List[Tuple[float, float]] = []
    cur: List[Tuple[float, float]] = [ordered_hits[0]]
    for i in range(1, len(ordered_hits)):
        px, py = ordered_hits[i - 1]
        cx, cy = ordered_hits[i]
        if abs(cx - px) <= jump_thresh and abs(cy - py) <= jump_thresh:
            cur.append((cx, cy))
        else:
            if len(cur) > best_len:
                best_len = len(cur)
                best_slice = list(cur)
            cur = [(cx, cy)]
    if len(cur) > best_len:
        best_slice = cur
    if len(best_slice) < max(3, len(ordered_hits) // 5):
        return None
    return (
        max(0.0, min(1.0, _median([h[0] for h in best_slice]))),
        max(0.0, min(1.0, _median([h[1] for h in best_slice]))),
    )


def detect_face_center_ratios_for_clip_samples(
    video_path: Path, clip: Clip
) -> Optional[Tuple[float, float]]:
    clip_start = float(clip.start)
    clip_end = float(clip.end)
    sample_count = 24
    hits_ordered: List[Tuple[float, float]] = []
    for t in _stratified_clip_sample_times(clip_start, clip_end, sample_count):
        raw = _ffmpeg_extract_frame_png_bytes(video_path, t)
        if not raw:
            continue
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except Exception:
            continue
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        m = _best_face_center_ratios(frame)
        if m:
            hits_ordered.append(m)
    if not hits_ordered:
        return None

    dom = _dominant_spatial_cluster_median(hits_ordered)
    run = _longest_temporal_run_median(hits_ordered)
    # Se a corrida longa concorda com o cluster dominante, reforça; se diverge muito,
    # confia mais no cluster (mais amostras espaciais).
    if run is not None:
        if abs(run[0] - dom[0]) <= 0.12 and abs(run[1] - dom[1]) <= 0.12:
            return (
                max(0.0, min(1.0, 0.5 * (dom[0] + run[0]))),
                max(0.0, min(1.0, 0.5 * (dom[1] + run[1]))),
            )
    return dom


def _combine_face_ratios(
    a: Optional[Tuple[float, float]],
    b: Optional[Tuple[float, float]],
    weight_a: float,
) -> Optional[Tuple[float, float]]:
    if a and b:
        w = max(0.0, min(1.0, weight_a))
        return (
            max(0.0, min(1.0, w * a[0] + (1.0 - w) * b[0])),
            max(0.0, min(1.0, w * a[1] + (1.0 - w) * b[1])),
        )
    return a or b


def _crop_vf_scale_increase_face(
    target_w: int, target_h: int, face_xy: Optional[Tuple[float, float]]
) -> str:
    """
    Após scale=increase, recorta target_w x target_h alinhado ao rosto em X e Y.
    Sem rosto, usa crop centrado (comportamento por omissão do FFmpeg).
    """
    scale = f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase"
    if face_xy is None:
        return f"{scale},crop={target_w}:{target_h}"
    cx, cy = face_xy
    tw, th = target_w, target_h
    x_expr = f"max(0\\,min(iw-{tw}\\,iw*{cx:.6f}-{tw/2:.2f}))"
    y_expr = f"max(0\\,min(ih-{th}\\,ih*{cy:.6f}-{th/2:.2f}))"
    return f"{scale},crop={tw}:{th}:{x_expr}:{y_expr}"


def build_framing_vf(
    target_w: int,
    target_h: int,
    framing_mode: str,
    video_path: Path,
    clip: Optional[Clip],
    focus_time_abs: Optional[float],
) -> str:
    lower = framing_mode.lower()
    if "inteligente" in lower or "rosto" in lower:
        r_focus: Optional[Tuple[float, float]] = None
        r_clip: Optional[Tuple[float, float]] = None
        if focus_time_abs is not None:
            r_focus = detect_face_center_ratios_at_time(video_path, focus_time_abs)
        if clip is not None:
            r_clip = detect_face_center_ratios_for_clip_samples(video_path, clip)
        # Um único instante (ex.: capa) não deve puxar o crop após cortes de câmara —
        # o agregado do clipe (cluster dominante) manda.
        face_xy = _combine_face_ratios(r_focus, r_clip, weight_a=0.06)
        return _crop_vf_scale_increase_face(target_w, target_h, face_xy)
    if "crop" in lower:
        return (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h}"
        )
    return (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
    )


def export_preview_frame_png_bytes(
    video_path: Path,
    *,
    resolution: str,
    export_quality: str,
    aspect_ratio: str,
    framing_mode: str,
    t_sec: float = 0.0,
) -> Optional[bytes]:
    """
    Primeiro frame (ou instante ``t_sec``) já com o mesmo enquadramento da exportação.
    Usado na pré-visualização da capa e das legendas na UI.
    """
    target_w, target_h = get_export_dimensions(resolution, export_quality, aspect_ratio)
    vf = build_framing_vf(
        target_w,
        target_h,
        framing_mode,
        video_path,
        clip=None,
        focus_time_abs=float(t_sec),
    )
    full_vf = f"setpts=PTS-STARTPTS,{vf}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, t_sec):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        full_vf,
        "-f",
        "image2pipe",
        "-c:v",
        "png",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode(errors="replace")[:500]
        logger.debug("export_preview_frame_png_bytes falhou: %s", err)
        return None
    return proc.stdout


def _path_for_ffmpeg_filter(path: Path) -> str:
    s = path.resolve().as_posix()
    if len(s) >= 2 and s[1] == ":":
        return s[0] + r"\:" + s[2:]
    return s


def create_social_cover(
    video_path: Path,
    output_dir: Path,
    clip_index: int,
    frame_second_abs: float,
    hook_phrase: str,
    *,
    resolution: str = "1080p",
    export_quality: str = "HD (1080p)",
    aspect_ratio: str = "Vertical (9:16) - Redes sociais",
    framing_mode: str = "Manter conteúdo (com bordas)",
    clip: Optional[Clip] = None,
    cover_style: Optional[SocialCoverStyle] = None,
) -> Path:
    """
    Cria capa JPG alinhada à resolução/formato da exportação e ao modo de enquadramento.
    Usa FFmpeg + texto (textfile) para evitar falhas com caracteres especiais no drawtext.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cover_path = output_dir / EXPORT_CLIP_COVER_FILENAME
    target_w, target_h = get_export_dimensions(resolution, export_quality, aspect_ratio)

    hook_line = (hook_phrase or "").strip().replace("\r", " ").replace("\n", " ")[:400]
    if not hook_line:
        hook_line = "Momento chave"

    base_vf = build_framing_vf(
        target_w,
        target_h,
        framing_mode,
        video_path,
        clip,
        focus_time_abs=float(frame_second_abs),
    )

    # Mesmo motor visual das legendas TikTok (ASS / libass), com quebra de linha segura.
    fs_base, _mv0 = tiktok_subtitle_style_sizes(target_w, target_h)
    if cover_style is not None and cover_style.font_size and cover_style.font_size > 0:
        fs = cover_style.font_size
    else:
        fs = fs_base
    max_chars = max(16, int(target_w / max(fs * 0.45, 1.0)))
    wrapped = textwrap.wrap(
        hook_line,
        width=max_chars,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if not wrapped:
        wrapped = [hook_line]
    hook_ass_body = r"\N".join(_ass_escape_basic(line) for line in wrapped)

    ass_cover_path = output_dir / f"_cap_overlay_{clip_index}.ass"
    ass_cover_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "ScriptType: v4.00+",
                f"PlayResX: {target_w}",
                f"PlayResY: {target_h}",
                "",
                "[V4+ Styles]",
                tiktok_cover_ass_v4_style_block(target_w, target_h, cover_style),
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                f"Dialogue: 0,0:00:00.00,0:00:30.00,Cover,,0,0,0,,{hook_ass_body}",
            ]
        ),
        encoding="utf-8",
    )
    ass_ff = _path_for_ffmpeg_filter(ass_cover_path)
    overlay_vf = f"{base_vf},ass='{ass_ff}'"

    def _run(cmd: list[str]) -> None:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            cwd=str(output_dir),
        )

    t = max(0.0, float(frame_second_abs))
    try:
        _run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{t:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                overlay_vf,
                str(cover_path),
            ]
        )
        return cover_path
    except subprocess.CalledProcessError as e:
        logger.warning(
            "Capa com texto falhou (clipe %s); tentando só enquadramento. FFmpeg: %s",
            clip_index,
            (e.stderr or "")[-500:],
        )
        try:
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{t:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-vf",
                    base_vf,
                    str(cover_path),
                ]
            )
            return cover_path
        except subprocess.CalledProcessError as e2:
            logger.warning(
                "Capa com enquadramento falhou (clipe %s); frame bruto. FFmpeg: %s",
                clip_index,
                (e2.stderr or "")[-500:],
            )
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{t:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(cover_path),
                ]
            )
            return cover_path
    finally:
        try:
            ass_cover_path.unlink(missing_ok=True)
        except OSError:
            pass


def extract_safe_audio(video_path: Path, output_dir: Path) -> Path:
    """
    Extrai um WAV mono leve a 16kHz para o Whisper.
    Equivalente ao bloco FFmpeg que gerava temp_audio_safe.wav.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_audio_path = output_dir / "temp_audio_safe.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(temp_audio_path),
    ]
    logger.info("Extraindo áudio seguro via FFmpeg...")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return temp_audio_path


def render_clips(
    video_path: Path,
    clips: Iterable[Clip],
    output_dir: Path,
    segments: List[Dict[str, Any]] | None = None,
    resolution: str = "1080p",
    export_quality: str = "Alta (mais lenta)",
    aspect_ratio: str = "Vertical (9:16) - Redes sociais",
    framing_mode: str = "Manter conteúdo (com bordas)",
    enable_tiktok_captions: bool = False,
    bitrate: str | None = None,
    tiktok_caption_style: Optional[TiktokCaptionStyle] = None,
    enable_moviepy_engagement: bool = False,
    *,
    on_clip_progress: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Renderiza uma lista de clipes para MP4 H.264, mantendo a mesma lógica do código original.
    ``on_clip_progress``: chamado após cada clipe com (índice atual, total), índice base 1.
    ``enable_moviepy_engagement``: segundo passe com MoviePy (push-in e fades só no vídeo) —
    exige ``pip install moviepy``; se o pacote faltar, regista aviso e mantém só o encode FFmpeg.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_list = list(clips)
    n_clips = len(clip_list)
    if n_clips == 0:
        return

    profile = PROFILE_MAP.get(export_quality) or PROFILE_MAP.get(resolution) or FALLBACK_PROFILE
    target_w, target_h = get_export_dimensions(resolution, export_quality, aspect_ratio)

    # Usa GPU (NVENC) se disponível; caso contrário mantém libx264.
    use_nvenc = _ffmpeg_has_encoder("h264_nvenc")
    v_encoder = "h264_nvenc" if use_nvenc else "libx264"
    if use_nvenc:
        logger.info("FFmpeg: encoder GPU detectado (h264_nvenc). Renderização acelerada ativada.")

    def sec_to_ass(ts: float) -> str:
        ts = max(0.0, ts)
        total_cs = int(round(ts * 100))
        hours = total_cs // 360000
        rem = total_cs % 360000
        minutes = rem // 6000
        rem = rem % 6000
        seconds = rem // 100
        centiseconds = rem % 100
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    def ass_escape(text: str) -> str:
        return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")

    def build_tiktok_ass_for_clip(
        clip: Clip, clip_index: int, work_dir: Path
    ) -> Path | None:
        if not segments:
            return None
        #TODO: Ajustar a legenda para o TikTok
        # Ajuste fino para evitar adiantamento residual da legenda.
        # Valor positivo atrasa levemente a exibição.
        subtitle_delay_s = 0.20
        words: List[Dict[str, Any]] = []
        for seg in segments:
            for w in seg.get("words") or []:
                ws = float(w.get("start", 0.0))
                we = float(w.get("end", 0.0))
                if we <= clip.start or ws >= clip.end:
                    continue
                words.append(
                    {
                        "start": max(ws, clip.start),
                        "end": min(we, clip.end),
                        "word": str(w.get("word", "")).strip(),
                    }
                )
        words = [w for w in words if w["word"] and w["end"] > w["start"]]

        ass_path = work_dir / "clipe_captions.ass"
        lines: List[str] = []
        if words:
            groups = [words[idx : idx + 4] for idx in range(0, len(words), 4)]
            for group in groups:
                start = max(0.0, (group[0]["start"] - clip.start) + subtitle_delay_s)
                end = max(start + 0.01, (group[-1]["end"] - clip.start) + subtitle_delay_s)
                karaoke_text = ""
                for idx, item in enumerate(group):
                    current_start = float(item["start"])
                    if idx < len(group) - 1:
                        # Usa o início da próxima palavra para manter pausas naturais
                        # e evitar adiantamento progressivo da legenda.
                        next_start = float(group[idx + 1]["start"])
                        dur_s = max(0.01, next_start - current_start)
                    else:
                        dur_s = max(0.01, float(item["end"]) - current_start)
                    dur_cs = max(1, int(round(dur_s * 100)))
                    karaoke_text += rf"{{\k{dur_cs}}}{ass_escape(item['word'])} "
                lines.append(
                    f"Dialogue: 0,{sec_to_ass(start)},{sec_to_ass(end)},Default,,0,0,0,,{karaoke_text.strip()}"
                )
        else:
            # Fallback: sem timestamps por palavra, usa frases/segmentos para não sair sem legenda.
            for seg in segments:
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", 0.0))
                if seg_end <= clip.start or seg_start >= clip.end:
                    continue
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                rel_start = max(0.0, max(seg_start, clip.start) - clip.start)
                rel_end = max(rel_start + 0.2, min(seg_end, clip.end) - clip.start)
                rel_start += subtitle_delay_s
                rel_end = max(rel_start + 0.2, rel_end + subtitle_delay_s)
                lines.append(
                    f"Dialogue: 0,{sec_to_ass(rel_start)},{sec_to_ass(rel_end)},Default,,0,0,0,,{ass_escape(text)}"
                )

        if not lines:
            logger.info("Clipe %s sem texto útil para legenda.", clip_index)
            return None
        logger.info("Clipe %s: %s linha(s) de legenda geradas.", clip_index, len(lines))

        ass_content = "\n".join(
            [
                "[Script Info]",
                "ScriptType: v4.00+",
                f"PlayResX: {target_w}",
                f"PlayResY: {target_h}",
                "",
                "[V4+ Styles]",
                tiktok_ass_v4_style_block(target_w, target_h, tiktok_caption_style),
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                *lines,
            ]
        )
        ass_path.write_text(ass_content, encoding="utf-8")
        return ass_path

    for i, clip in enumerate(clip_list, start=1):
        clip_dir = clip_session_subdirectory(output_dir, i)
        clip_dir.mkdir(parents=True, exist_ok=True)
        output_file = clip_dir / EXPORT_CLIP_VIDEO_FILENAME
        logger.info(
            "Renderizando clipe %s/%s → %s",
            i,
            n_clips,
            output_file,
        )

        ass_path: Path | None = None
        if enable_tiktok_captions:
            ass_path = build_tiktok_ass_for_clip(clip, i, clip_dir)
            if ass_path:
                logger.info("Legenda preparada para clipe %s (%s).", i, ass_path.name)
            else:
                logger.info("Legenda não aplicada no clipe %s (sem conteúdo de legenda).", i)

        # 1) FFmpeg: corte + enquadramento (+ eventual ficheiro base para legenda ou MoviePy).
        # 2) Opcional: MoviePy (retenção).
        # 3) Opcional: burn-in ASS no resultado final.
        needs_base_file = bool(ass_path) or bool(enable_moviepy_engagement)
        base_output_file = clip_dir / CLIP_BASE_FILENAME if needs_base_file else output_file

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ss",
            str(clip.start),
            "-to",
            str(clip.end),
            "-c:v",
            v_encoder,
            "-preset",
            profile["preset"],
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-fps_mode",
            "cfr",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            "aresample=async=1",
        ]
        # libx264 usa CRF; NVENC usa CQ/RC (mantemos default simples e só removemos CRF).
        if v_encoder == "libx264":
            cmd.extend(["-crf", profile["crf"]])
        else:
            # Qualidade visual aproximada ao CRF: usa CQ. (Valor moderado; mantém velocidade.)
            cmd.extend(["-cq", str(int(profile["crf"]))])

        framing_vf = build_framing_vf(
            target_w,
            target_h,
            framing_mode,
            video_path,
            clip,
            focus_time_abs=None,
        )
        filters: List[str] = ["setpts=PTS-STARTPTS", framing_vf]

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        if bitrate:
            # Substitui CRF por bitrate fixo
            if "-crf" in cmd:
                idx = cmd.index("-crf")
                cmd[idx] = "-b:v"
                cmd[idx + 1] = f"{bitrate}k"

        cmd.append(str(base_output_file))

        try:
            base_proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
                cwd=str(clip_dir),
            )
        except subprocess.CalledProcessError as e:
            err_tail = "\n".join((e.stderr or "").splitlines()[-25:])
            raise RuntimeError(f"FFmpeg falhou ao renderizar base do clipe {i}.\n{err_tail}") from e

        post_path = base_output_file
        if enable_moviepy_engagement:
            from app.services.engagement_effects_catalog import resolve_moviepy_effect_ids
            from app.services.moviepy_engagement import MOVIEPY_AVAILABLE, apply_engagement_effects

            engagement_out = clip_dir / CLIP_MOVIEPY_TEMP
            skip_mx = resolve_moviepy_effect_ids(clip.engagement_effects) is None
            if skip_mx:
                logger.info(
                    "Clipe %s/%s: efeitos MoviePy omitidos (engagement_effects pede 'none').",
                    i,
                    n_clips,
                )
            elif MOVIEPY_AVAILABLE:
                logger.info(
                    "Clipe %s/%s: pós-processamento MoviePy (push-in e/ou fade no vídeo)...",
                    i,
                    n_clips,
                )
                ok_mx = apply_engagement_effects(
                    Path(post_path),
                    engagement_out,
                    crf=str(profile["crf"]),
                    preset=str(profile["preset"]),
                    engagement_effects=clip.engagement_effects,
                )
                if ok_mx:
                    if (
                        Path(post_path).resolve() != engagement_out.resolve()
                        and Path(post_path).exists()
                    ):
                        Path(post_path).unlink(missing_ok=True)
                    post_path = engagement_out
                else:
                    engagement_out.unlink(missing_ok=True)
            else:
                logger.warning(
                    "Efeitos de retenção (MoviePy) ligados mas o pacote não está instalado "
                    "(pip install moviepy). Clipe %s: a ignorar este passo.",
                    i,
                )

        proc = base_proc
        if ass_path:
            ass_for_ffmpeg = ass_path.name.replace("'", r"\'")
            subtitle_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(post_path),
                "-vf",
                f"subtitles=filename='{ass_for_ffmpeg}'",
                "-c:v",
                v_encoder,
                "-preset",
                profile["preset"],
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                str(output_file),
            ]
            if v_encoder == "libx264":
                subtitle_cmd.extend(["-crf", profile["crf"]])
            else:
                subtitle_cmd.extend(["-cq", str(int(profile["crf"]))])
            try:
                proc = subprocess.run(
                    subtitle_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                    cwd=str(clip_dir),
                )
            except subprocess.CalledProcessError:
                # Fallback para builds do FFmpeg onde subtitles falha com .ass.
                fallback_cmd = list(subtitle_cmd)
                vf_idx = fallback_cmd.index("-vf")
                fallback_cmd[vf_idx + 1] = f"ass='{ass_for_ffmpeg}'"
                try:
                    proc = subprocess.run(
                        fallback_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=True,
                        cwd=str(clip_dir),
                    )
                except subprocess.CalledProcessError as e2:
                    err_tail = "\n".join((e2.stderr or "").splitlines()[-25:])
                    raise RuntimeError(
                        f"FFmpeg falhou ao aplicar legenda no clipe {i} (subtitles e ass).\n{err_tail}"
                    ) from e2
            if (
                Path(post_path).exists()
                and Path(post_path).resolve() != output_file.resolve()
            ):
                Path(post_path).unlink(missing_ok=True)
        else:
            if Path(post_path).resolve() != output_file.resolve():
                if output_file.exists():
                    output_file.unlink(missing_ok=True)
                Path(post_path).replace(output_file)

        if enable_tiktok_captions and ass_path:
            ffmpeg_log = (proc.stderr or "").strip()
            if ffmpeg_log:
                tail = "\n".join(ffmpeg_log.splitlines()[-12:])
                logger.info("FFmpeg (clipe %s) log final:\n%s", i, tail)
        if ass_path and ass_path.exists():
            ass_path.unlink(missing_ok=True)

        if on_clip_progress is not None:
            on_clip_progress(i, n_clips)
        logger.info("Clipe %s/%s concluído.", i, n_clips)

