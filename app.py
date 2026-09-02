from __future__ import annotations

import csv
import hashlib
import io
import json
import mimetypes
import os
import platform
import re
import smtplib
import zipfile
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

from segmentation import colorize_regions, coverage_percent, prepare_multisegment, segment_image


APP_TITLE = "满浆率计算"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROFILE_DIR = Path("temp/profiles")
MAX_QUEUE_IMAGES = 200
MAX_ARCHIVE_MEMBER_BYTES = 80 * 1024 * 1024

st.set_page_config(page_title=APP_TITLE, page_icon="🧱", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background:#f7f9fc;border:1px solid #e5eaf1;border-radius:14px;padding:12px 16px;}
      [data-testid="stSidebar"] hr {margin: 1rem 0;}
      .queue-hint {color:#607083;font-size:.88rem;line-height:1.55;}
      .step-strip {padding:.7rem 1rem;border-radius:12px;background:linear-gradient(90deg,#edf7ff,#f5fbf8);border:1px solid #d9ebf5;margin-bottom:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _apply_pending_profile() -> None:
    pending = st.session_state.pop("pending_multisegment_profile", None)
    if not pending:
        return
    max_regions = int(pending.get("max_regions", 6))
    mode = pending.get("application_mode", "fixed")
    st.session_state["multi_max_regions"] = max_regions
    st.session_state["multi_application_mode"] = "固定阈值" if mode == "fixed" else "相对自动阈值偏移"
    prefix = "multi_fixed" if mode == "fixed" else "multi_relative"
    for region_id, value in pending.get("region_values", {}).items():
        st.session_state[f"{prefix}_{max_regions}_{region_id}"] = int(round(float(value)))
    st.session_state["profile_name"] = pending.get("name", "默认多段标准")
    st.session_state["profile_loaded_notice"] = pending.get("name", "参数标准")


_apply_pending_profile()
for state_key, default_value in {"selected_image_id": None, "analysis_bundle": None}.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


def _decode_image(data: bytes) -> np.ndarray | None:
    cache = st.session_state.setdefault("_decoded_image_cache", {})
    cache_key = hashlib.sha1(data).hexdigest()
    if cache_key not in cache:
        cache[cache_key] = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        while len(cache) > 24:
            cache.pop(next(iter(cache)))
    return cache[cache_key]


def _prepare_multi_cached(data: bytes, max_regions: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    cache = st.session_state.setdefault("_multisegment_preparation_cache", {})
    cache_key = f"{hashlib.sha1(data).hexdigest()}:{max_regions}"
    if cache_key in cache:
        return cache[cache_key]
    image = _decode_image(data)
    if image is None:
        raise ValueError("无法解码图像")
    prepared = prepare_multisegment(image, max_regions=max_regions)
    cache[cache_key] = (prepared.normalized, prepared.region_map, prepared.automatic_thresholds)
    while len(cache) > 16:
        cache.pop(next(iter(cache)))
    return cache[cache_key]


def _unique_name(name: str, used: set[str]) -> str:
    clean = Path(name).name or "image"
    stem, suffix = os.path.splitext(clean)
    candidate = clean
    index = 2
    while candidate.lower() in used:
        candidate = f"{stem} ({index}){suffix}"
        index += 1
    used.add(candidate.lower())
    return candidate


def build_image_queue(uploaded_files: list[Any] | None) -> tuple[list[dict[str, Any]], list[str]]:
    queue: list[dict[str, Any]] = []
    warnings: list[str] = []
    used_names: set[str] = set()

    def append_image(name: str, data: bytes, source: str) -> None:
        if len(queue) >= MAX_QUEUE_IMAGES:
            return
        image = _decode_image(data)
        if image is None:
            warnings.append(f"{name} 无法识别，已跳过。")
            return
        display_name = _unique_name(name, used_names)
        image_id = hashlib.sha1(display_name.encode("utf-8") + data[:65536]).hexdigest()
        queue.append({"id": image_id, "name": display_name, "source": source, "data": data, "image": image.copy()})

    for uploaded in uploaded_files or []:
        data = uploaded.getvalue()
        suffix = Path(uploaded.name).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            append_image(uploaded.name, data, "直接上传")
            continue
        if suffix != ".zip":
            warnings.append(f"{uploaded.name} 格式不受支持，已跳过。")
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.infolist():
                    if len(queue) >= MAX_QUEUE_IMAGES:
                        warnings.append(f"队列最多保留 {MAX_QUEUE_IMAGES} 张图像，其余文件已跳过。")
                        break
                    if member.is_dir() or Path(member.filename).suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    if member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                        warnings.append(f"{member.filename} 超过单图 80 MiB 限制，已跳过。")
                        continue
                    append_image(member.filename, archive.read(member), f"{uploaded.name} / ZIP")
        except (zipfile.BadZipFile, RuntimeError) as exc:
            warnings.append(f"{uploaded.name} 读取失败：{exc}")
    return queue, warnings


def _safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", value.strip()).strip("-")
    return value[:48] or "默认多段标准"


def _profile_files() -> list[Path]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(PROFILE_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def _load_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("不支持的参数文件版本")
    return payload


def _save_profile(payload: dict[str, Any]) -> Path:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILE_DIR / f"{_safe_name(payload['name'])}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _segment(image: np.ndarray, config: dict[str, Any]):
    return segment_image(
        image,
        config["method"],
        foreground_dark=config["foreground_dark"],
        global_threshold=config["global_threshold"],
        adaptive_block=config["adaptive_block"],
        adaptive_c=config["adaptive_c"],
        max_regions=config["max_regions"],
        region_values=config["region_values"],
        application_mode=config["application_mode"],
        kernel_size=config["kernel_size"],
        min_area=config["min_area"],
    )


def _segment_cached(data: bytes, config: dict[str, Any]):
    cache = st.session_state.setdefault("_segmentation_result_cache", {})
    cache_key = f"{hashlib.sha1(data).hexdigest()}:{_config_fingerprint(config)}"
    if cache_key not in cache:
        image = _decode_image(data)
        if image is None:
            raise ValueError("无法解码图像")
        cache[cache_key] = _segment(image, config)
        while len(cache) > 24:
            cache.pop(next(iter(cache)))
    return cache[cache_key]


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    colored = image.copy()
    colored[mask > 0] = (62, 201, 118)
    return cv2.addWeighted(image, 0.60, colored, 0.40, 0)


def _fit_for_display(image: np.ndarray, max_width: int, convert_bgr: bool = False) -> np.ndarray:
    height, width = image.shape[:2]
    shown = image
    if width > max_width:
        scale = max_width / width
        shown = cv2.resize(image, (max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    if convert_bgr:
        shown = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
    return shown


def _result_board(image: np.ndarray, mask: np.ndarray, rate: float, name: str, description: str) -> np.ndarray:
    height, width = image.shape[:2]
    header_height = max(72, min(120, height // 8))
    board = np.full((height + header_height, width * 2, 3), 24, dtype=np.uint8)
    board[header_height:, :width] = image
    board[header_height:, width:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(board, f"Slurry rate  {rate:.2f}%", (24, int(header_height * 0.48)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.65, min(1.25, width / 900)), (111, 232, 178), 2, cv2.LINE_AA)
    subtitle = f"{description}  |  {name}  |  Original / Foreground mask"
    cv2.putText(board, subtitle[:120], (24, int(header_height * 0.80)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.40, min(0.70, width / 1500)), (220, 226, 235), 1, cv2.LINE_AA)
    cv2.line(board, (width, header_height), (width, height + header_height), (235, 235, 235), 2)
    return board


def _encode_image(image: np.ndarray, extension: str = ".jpg") -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, 94] if extension == ".jpg" else []
    ok, buffer = cv2.imencode(extension, image, params)
    if not ok:
        raise ValueError("图像编码失败")
    return buffer.tobytes()


def _result_csv(results: list[dict[str, Any]]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["文件名", "状态", "满浆率(%)", "算法"])
    for result in results:
        writer.writerow([result["name"], result["status"], result.get("rate", ""), result["algorithm"]])
    return output.getvalue().encode("utf-8-sig")


def _result_zip(paths: list[str]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for raw_path in paths:
            path = Path(raw_path)
            if path.is_file():
                archive.write(path, path.name)
    return output.getvalue()


def process_queue(queue: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("temp") / f"batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0.0)
    status = st.empty()
    results: list[dict[str, Any]] = []
    saved_files: list[str] = []

    parameters_path = output_dir / "analysis_parameters.json"
    parameters_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    saved_files.append(str(parameters_path))
    for index, item in enumerate(queue, start=1):
        status.info(f"[{index}/{len(queue)}] 正在分析：{item['name']}")
        try:
            segmented = _segment_cached(item["data"], config)
            rate = coverage_percent(segmented.mask)
            stem = _safe_name(Path(item["name"]).stem)
            result_image = _result_board(item["image"], segmented.mask, rate, item["name"], config["description"])
            overlay = _overlay(item["image"], segmented.mask)
            result_path = output_dir / f"{stem}_result.jpg"
            overlay_path = output_dir / f"{stem}_overlay.jpg"
            mask_path = output_dir / f"{stem}_mask.png"
            cv2.imwrite(str(result_path), result_image)
            cv2.imwrite(str(overlay_path), overlay)
            cv2.imwrite(str(mask_path), segmented.mask)
            saved_files.extend([str(result_path), str(overlay_path), str(mask_path)])
            results.append({"id": item["id"], "name": item["name"], "status": "完成", "rate": round(rate, 2),
                            "algorithm": config["algorithm_label"], "mask": segmented.mask, "result_path": str(result_path)})
        except Exception as exc:
            results.append({"id": item["id"], "name": item["name"], "status": f"失败：{exc}", "rate": None,
                            "algorithm": config["algorithm_label"], "mask": None})
        progress.progress(index / len(queue))
    succeeded = sum(row["status"] == "完成" for row in results)
    status.success(f"队列分析完成：{succeeded}/{len(queue)} 张成功")
    return {"fingerprint": _config_fingerprint(config), "timestamp": timestamp, "output_dir": str(output_dir),
            "results": results, "saved_files": saved_files}


def _height_map(masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_height, first_width = masks[0].shape
    scale = min(1.0, 150.0 / max(first_height, first_width))
    target_width = max(24, int(round(first_width * scale)))
    target_height = max(24, int(round(first_height * scale)))
    hits = np.zeros((target_height, target_width), dtype=np.uint16)
    for mask in masks:
        resized = cv2.resize((mask > 0).astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        hits += resized
    return np.linspace(0, 100, target_width), np.linspace(0, 100, target_height), hits


def render_3d_visualization(results: list[dict[str, Any]]) -> None:
    valid = [result for result in results if result.get("mask") is not None]
    if len(valid) < 2 or not st.toggle("查看 3D 前景命中高程", value=False, help="至少两张图完成分析后可用"):
        return
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("缺少 Plotly 依赖，请重新构建部署镜像。")
        return
    x, y, hits = _height_map([result["mask"] for result in valid])
    hit_rate = hits.astype(np.float32) / len(valid) * 100.0
    colorscale = [[0.00, "#0b1739"], [0.18, "#174ea6"], [0.42, "#15a7c8"],
                  [0.68, "#48c78e"], [0.86, "#f0c95a"], [1.00, "#f36b45"]]
    surface = go.Figure(data=[go.Surface(
        x=x, y=y, z=hits, surfacecolor=hit_rate, colorscale=colorscale, cmin=0, cmax=100,
        customdata=hit_rate, colorbar=dict(title="命中率 %", thickness=16, len=0.72),
        hovertemplate="横向 %{x:.1f}%<br>纵向 %{y:.1f}%<br>命中 %{z} 张<br>命中率 %{customdata:.1f}%<extra></extra>",
        contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
        lighting=dict(ambient=0.65, diffuse=0.75, specular=0.24, roughness=0.72))])
    surface.update_layout(
        title=dict(text=f"前景稳定性高程 · {len(valid)} 张图像", x=0.03), height=670,
        margin=dict(l=0, r=0, t=65, b=0), paper_bgcolor="#ffffff",
        scene=dict(bgcolor="#f6f8fb", xaxis=dict(title="图像横向位置 (%)", gridcolor="#dfe5ec"),
                   yaxis=dict(title="图像纵向位置 (%)", gridcolor="#dfe5ec", autorange="reversed"),
                   zaxis=dict(title="前景命中张数", range=[0, len(valid)], dtick=max(1, len(valid) // 5)),
                   camera=dict(eye=dict(x=1.45, y=-1.55, z=1.15)), aspectmode="manual",
                   aspectratio=dict(x=1.45, y=1.0, z=0.62)))
    top_view = go.Figure(data=go.Heatmap(
        x=x, y=y, z=hit_rate, colorscale=colorscale, zmin=0, zmax=100, colorbar=dict(title="命中率 %"),
        hovertemplate="横向 %{x:.1f}%<br>纵向 %{y:.1f}%<br>命中率 %{z:.1f}%<extra></extra>"))
    top_view.update_layout(title="俯视命中热力图", height=560, margin=dict(l=40, r=20, t=55, b=45),
                           xaxis_title="图像横向位置 (%)", yaxis_title="图像纵向位置 (%)", yaxis_autorange="reversed")
    tab_3d, tab_heat = st.tabs(["3D 高程", "俯视热力图"])
    with tab_3d:
        st.plotly_chart(surface, use_container_width=True, config={"displaylogo": False})
    with tab_heat:
        st.plotly_chart(top_view, use_container_width=True, config={"displaylogo": False})
    st.caption("高程表示该位置被判定为前景的图像张数，颜色表示命中率。不同尺寸图像按归一化坐标对齐；只有拍摄视角和裁剪范围基本一致时，逐像素比较才有业务意义。")


def _get_smtp_config() -> dict[str, Any] | None:
    try:
        config = st.secrets.get("smtp", None)
    except Exception:
        config = None
    if not config or not all((config.get("host"), config.get("user"), config.get("password"))):
        return None
    return {"host": config["host"], "port": int(config.get("port", 587)), "user": config["user"],
            "password": config["password"], "from": config.get("from", config["user"]), "tls": bool(config.get("tls", True))}


def _record_feedback(kind: str, name: str, email: str, message: str, attachments: list[Any]) -> tuple[bool, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feedback_dir = Path("temp/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    log_path = feedback_dir / "feedback_log.csv"
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if log_path.stat().st_size == 0:
            writer.writerow(["timestamp", "type", "name", "email", "message", "platform", "python", "streamlit"])
        writer.writerow([timestamp, kind, name, email, message, platform.platform(), platform.python_version(), st.__version__])
    prepared: list[tuple[str, bytes]] = []
    for index, uploaded in enumerate(attachments or [], start=1):
        data = uploaded.getvalue()
        filename = f"{timestamp}_{index}_{Path(uploaded.name).name}"
        (feedback_dir / filename).write_bytes(data)
        prepared.append((uploaded.name, data))
    smtp = _get_smtp_config()
    if not smtp:
        return False, "反馈已保存到服务器；SMTP 未配置，因此未发送邮件。"
    try:
        mail = EmailMessage()
        mail["Subject"], mail["From"], mail["To"] = f"满浆率工具使用反馈 - {kind}", smtp["from"], "guozhu_l@163.com"
        mail.set_content(f"称呼：{name}\n邮箱：{email}\n\n{message}")
        for filename, data in prepared:
            content_type, _ = mimetypes.guess_type(filename)
            main_type, sub_type = (content_type or "application/octet-stream").split("/", 1)
            mail.add_attachment(data, maintype=main_type, subtype=sub_type, filename=filename)
        with smtplib.SMTP(smtp["host"], smtp["port"], timeout=12) as server:
            if smtp["tls"]:
                server.starttls()
            server.login(smtp["user"], smtp["password"])
            server.send_message(mail)
        return True, "反馈邮件已发送。"
    except Exception as exc:
        return False, f"反馈已保存到服务器；邮件发送失败：{exc}"


# Page heading
title_col, manual_col = st.columns([4, 1])
with title_col:
    st.title(APP_TITLE)
    st.caption("统一图像队列 · 参数标准化 · 批量分析 · 多图 3D 稳定性观察")
with manual_col:
    st.write("")
    show_manual = st.toggle("📖 使用手册", value=False)
if show_manual:
    try:
        with st.expander("使用手册", expanded=True):
            st.markdown(Path("user_manual.md").read_text(encoding="utf-8"))
    except OSError as exc:
        st.warning(f"使用手册读取失败：{exc}")
st.markdown('<div class="step-strip">① 上传图像形成队列　→　② 选择自动或高级算法　→　③ 检查当前图预览　→　④ 分析整个队列　→　⑤ 下载结果或查看多图 3D</div>', unsafe_allow_html=True)

# Unified sidebar
st.sidebar.header("图像输入")
uploaded_files = st.sidebar.file_uploader(
    "上传图像或 ZIP", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "zip"], accept_multiple_files=True,
    help="可一次选择一张、多张图片，也可把图片 ZIP 与普通图片一起加入同一队列。")
st.sidebar.caption("一个入口即可：1 张直接分析，多张自动进入图像队列并支持 3D 统计。")
queue, queue_warnings = build_image_queue(uploaded_files)
if queue:
    st.sidebar.success(f"当前队列：{len(queue)} 张")
else:
    st.sidebar.info("尚未上传图像")

st.sidebar.divider()
st.sidebar.header("算法")
algorithm_mode = st.sidebar.radio("算法模式", ["自动", "高级"], index=0, horizontal=True)
method, algorithm_label = "otsu", "Otsu 自动阈值"
global_threshold, adaptive_block, adaptive_c, max_regions = 160, 51, 3, 6
application_mode, region_values, kernel_size, min_area = "fixed", {}, 5, None
if algorithm_mode == "自动":
    automatic_scheme = st.sidebar.selectbox("自动方案", ["Otsu 自动阈值", "多段式自动分割"],
        help="光照均匀时优先使用 Otsu；存在明显阴影或亮度分区时可选择多段式。")
    if automatic_scheme == "多段式自动分割":
        method, algorithm_label = "multisegment", automatic_scheme
        st.sidebar.caption("自动完成光照归一化、最多 6 个不规则分区及每区 Otsu。")
    else:
        st.sidebar.caption("无需调节阈值，适合光照与背景较均匀的图像。")
else:
    advanced_algorithm = st.sidebar.selectbox("高级算法", ["全局阈值", "自适应阈值", "Otsu 自动阈值", "多段式分割"])
    method = {"全局阈值": "global", "自适应阈值": "adaptive", "Otsu 自动阈值": "otsu", "多段式分割": "multisegment"}[advanced_algorithm]
    algorithm_label = advanced_algorithm
    if method == "global":
        global_threshold = st.sidebar.slider("灰度阈值", 0, 255, 160)
    elif method == "adaptive":
        adaptive_block = st.sidebar.slider("局部窗口", 3, 151, 51, step=2)
        adaptive_c = st.sidebar.slider("局部偏移 C", -30, 30, 3)
    elif method == "multisegment":
        max_regions = st.sidebar.slider("最大分区数", 2, 6, 6, key="multi_max_regions")
        mode_label = st.sidebar.radio("批处理应用方式", ["固定阈值", "相对自动阈值偏移"], key="multi_application_mode",
            help="固定阈值强调同一标准；相对偏移会先在每张图自动计算，再应用相同修正量。")
        application_mode = "fixed" if mode_label == "固定阈值" else "relative"
        profiles = _profile_files()
        if profiles:
            profile_by_name = {path.stem: path for path in profiles}
            selected_profile = st.sidebar.selectbox("已保存参数标准", list(profile_by_name))
            if st.sidebar.button("加载所选标准", use_container_width=True):
                try:
                    st.session_state["pending_multisegment_profile"] = _load_profile(profile_by_name[selected_profile])
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"加载失败：{exc}")
    with st.sidebar.expander("后处理参数"):
        kernel_size = st.slider("平滑核尺寸", 3, 11, 5, step=2)
        min_area = st.number_input("最小前景连通区（像素）", min_value=1, max_value=500000, value=64, step=16)

tile_type = st.sidebar.selectbox("材料对比", ["黑胶白砖", "白胶黑砖"])
foreground_dark = tile_type == "黑胶白砖"
description = st.sidebar.text_input("测试项描述", value="满浆率检测")
st.sidebar.divider()
with st.sidebar.expander("📸 拍摄与队列提示", expanded=True):
    st.markdown("- 镜头尽量与砖面平行，避免透视变形。\n- 裁剪范围、方向和分辨率尽量一致。\n- 保证光线充足，避免硬阴影和反光。\n- 3D 命中统计要求多张图的空间位置可比较。")

for warning in queue_warnings:
    st.warning(warning)

if not queue:
    st.info("请从左侧上传一张或多张图片。系统会自动建立纵向图像队列，不再要求选择‘单图/批量’入口。")
else:
    valid_ids = {item["id"] for item in queue}
    if st.session_state["selected_image_id"] not in valid_ids:
        st.session_state["selected_image_id"] = queue[0]["id"]
    selected = next(item for item in queue if item["id"] == st.session_state["selected_image_id"])
    if "profile_loaded_notice" in st.session_state:
        st.success(f"已加载参数标准：{st.session_state.pop('profile_loaded_notice')}")

    if method == "multisegment" and algorithm_mode == "高级":
        _, selected_regions, automatic_thresholds = _prepare_multi_cached(selected["data"], max_regions)
        region_count = len(automatic_thresholds)
        with st.expander("多段式分区参数标准", expanded=True):
            st.caption("分区按光照残差由暗到亮编号。每个分区可独立设定阈值；保存后，整队分析和以后上传的图像都会按相同编号复用。")
            refresh_col, note_col = st.columns([1, 3])
            with refresh_col:
                use_auto = st.button("用当前图自动值填充", use_container_width=True)
            with note_col:
                st.caption(f"当前参考图：{selected['name']} · 实际识别 {region_count} 个分区")
            prefix = "multi_fixed" if application_mode == "fixed" else "multi_relative"
            if use_auto:
                for region_id, auto_value in enumerate(automatic_thresholds):
                    st.session_state[f"{prefix}_{max_regions}_{region_id}"] = int(round(auto_value)) if application_mode == "fixed" else 0
                st.rerun()
            region_values = {}
            control_columns = st.columns(2)
            for region_id, automatic in enumerate(automatic_thresholds):
                key = f"{prefix}_{max_regions}_{region_id}"
                default = int(round(automatic)) if application_mode == "fixed" else 0
                if key not in st.session_state:
                    st.session_state[key] = default
                with control_columns[region_id % 2]:
                    if application_mode == "fixed":
                        value = st.slider(f"分区 {region_id + 1} · 固定阈值", 0, 255, key=key,
                            help=f"当前参考图自动阈值：{automatic:.1f}")
                        st.caption(f"参考自动值 {automatic:.1f} · 实际使用 {value}")
                    else:
                        value = st.slider(f"分区 {region_id + 1} · 阈值偏移", -60, 60, key=key,
                            help=f"每张图先计算自动阈值，再叠加该偏移；参考自动值 {automatic:.1f}")
                        st.caption(f"参考自动值 {automatic:.1f} · 参考实际值 {np.clip(automatic + value, 0, 255):.1f}")
                    region_values[region_id] = float(value)
            profile_name = st.text_input("参数标准名称", value="默认多段标准", key="profile_name")
            profile_payload = {"schema_version": 1, "name": _safe_name(profile_name),
                "created_or_updated_at": datetime.now().isoformat(timespec="seconds"), "max_regions": max_regions,
                "application_mode": application_mode, "region_values": {str(key): value for key, value in region_values.items()},
                "reference_image": selected["name"], "foreground_dark": foreground_dark}
            save_profile_col, download_profile_col = st.columns(2)
            with save_profile_col:
                if st.button("保存/更新参数标准", type="primary", use_container_width=True):
                    st.success(f"已保存：{_save_profile(profile_payload)}")
            with download_profile_col:
                st.download_button("下载参数 JSON", json.dumps(profile_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"{profile_payload['name']}.json", mime="application/json", use_container_width=True)

    config = {"algorithm_mode": algorithm_mode, "algorithm_label": algorithm_label, "method": method,
        "foreground_dark": foreground_dark, "tile_type": tile_type, "global_threshold": int(global_threshold),
        "adaptive_block": int(adaptive_block), "adaptive_c": int(adaptive_c), "max_regions": int(max_regions),
        "application_mode": application_mode, "region_values": region_values, "kernel_size": int(kernel_size),
        "min_area": int(min_area) if min_area is not None else None, "description": description}
    fingerprint = _config_fingerprint(config)
    preview = _segment_cached(selected["data"], config)
    preview_rate = coverage_percent(preview.mask)

    action_col, count_col, algorithm_col, rate_col = st.columns([1.45, 0.8, 1.4, 0.8])
    with action_col:
        analyze_clicked = st.button(f"▶ 分析整个队列（{len(queue)} 张）", type="primary", use_container_width=True)
    with count_col:
        st.metric("队列", f"{len(queue)} 张")
    with algorithm_col:
        st.metric("当前算法", algorithm_label)
    with rate_col:
        st.metric("当前图预览", f"{preview_rate:.2f}%")
    if analyze_clicked:
        st.session_state["analysis_bundle"] = process_queue(queue, config)

    bundle = st.session_state.get("analysis_bundle")
    current_results = bool(bundle and bundle.get("fingerprint") == fingerprint)
    result_by_id = {result["id"]: result for result in (bundle.get("results", []) if current_results else [])}
    if bundle and not current_results:
        st.warning("算法或参数已经变化，下面缓存的是旧结果。请重新分析整个队列后再进行 3D 对比或下载。")

    queue_column, preview_column = st.columns([1.05, 3.4], gap="large")
    with queue_column:
        st.subheader("纵向图像浏览器")
        st.markdown('<div class="queue-hint">点击任一缩略图切换当前预览；整队分析始终使用同一套参数。</div>', unsafe_allow_html=True)
        for index, item in enumerate(queue, start=1):
            is_selected = item["id"] == selected["id"]
            st.image(_fit_for_display(item["image"], 360, convert_bgr=True), use_column_width=True)
            result = result_by_id.get(item["id"])
            st.button(f"{'●' if is_selected else '○'} {index}. {item['name']}", key=f"select_{item['id']}",
                use_container_width=True, on_click=lambda image_id=item["id"]: st.session_state.update(selected_image_id=image_id))
            if result and result.get("rate") is not None:
                st.caption(f"满浆率 {result['rate']:.2f}% · 已完成")
            else:
                height, width = item["image"].shape[:2]
                st.caption(f"{width} × {height} · {item['source']}")
            st.divider()

    with preview_column:
        st.subheader(f"当前图像 · {selected['name']}")
        compare_tab, overlay_tab, details_tab = st.tabs(["原图与掩码", "前景叠加", "分析详情"])
        with compare_tab:
            original_col, mask_col = st.columns(2)
            with original_col:
                st.image(_fit_for_display(selected["image"], 1200, convert_bgr=True), caption="原图", use_column_width=True)
            with mask_col:
                st.image(_fit_for_display(preview.mask, 1200), caption="前景掩码（白色计入满浆）", use_column_width=True, clamp=True)
        with overlay_tab:
            st.image(_fit_for_display(_overlay(selected["image"], preview.mask), 1200, convert_bgr=True),
                caption="绿色区域为当前识别前景", use_column_width=True)
        with details_tab:
            detail_col_1, detail_col_2, detail_col_3 = st.columns(3)
            detail_col_1.metric("满浆率", f"{preview_rate:.2f}%")
            detail_col_2.metric("前景像素", f"{np.count_nonzero(preview.mask):,}")
            detail_col_3.metric("总像素", f"{preview.mask.size:,}")
            if preview.region_map is not None:
                region_rgb = colorize_regions(preview.region_map)
                region_blend = cv2.addWeighted(cv2.cvtColor(selected["image"], cv2.COLOR_BGR2RGB), 0.56, region_rgb, 0.44, 0)
                st.image(_fit_for_display(region_blend, 1200), caption="多段式不规则分区", use_column_width=True)
                region_rows = []
                for region_id, applied in enumerate(preview.applied_thresholds or []):
                    selector = preview.region_map == region_id
                    region_rows.append({"分区": region_id + 1, "面积占比": f"{np.mean(selector) * 100:.2f}%",
                        "自动阈值": f"{preview.automatic_thresholds[region_id]:.1f}", "实际阈值": f"{applied:.1f}",
                        "分区前景率": f"{np.mean(preview.mask[selector] > 0) * 100:.2f}%"})
                st.dataframe(region_rows, use_container_width=True, hide_index=True)
        current_board = _result_board(selected["image"], preview.mask, preview_rate, selected["name"], description)
        current_stem = _safe_name(Path(selected["name"]).stem)
        download_col, save_col = st.columns(2)
        with download_col:
            st.download_button("📥 下载当前结果图", _encode_image(current_board), file_name=f"{current_stem}_result.jpg",
                mime="image/jpeg", use_container_width=True)
        with save_col:
            if st.button("💾 保存当前结果到服务器", use_container_width=True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = Path("temp") / f"{current_stem}_result_{timestamp}.jpg"
                cv2.imwrite(str(path), current_board)
                st.success(f"已保存：{path}")

    if current_results:
        st.divider()
        st.subheader("队列分析结果")
        table_rows = [{"文件名": result["name"], "状态": result["status"], "满浆率(%)": result.get("rate"),
                       "算法": result["algorithm"]} for result in bundle["results"]]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        csv_col, zip_col, clear_col = st.columns(3)
        with csv_col:
            st.download_button("📄 下载结果 CSV", _result_csv(bundle["results"]),
                file_name=f"batch_results_{bundle['timestamp']}.csv", mime="text/csv", use_container_width=True)
        with zip_col:
            st.download_button("📦 下载全部结果 ZIP", _result_zip(bundle["saved_files"]),
                file_name=f"batch_outputs_{bundle['timestamp']}.zip", mime="application/zip", type="primary", use_container_width=True)
        with clear_col:
            if st.button("清空分析结果", use_container_width=True):
                st.session_state["analysis_bundle"] = None
                st.rerun()
        if len(queue) >= 2:
            st.divider()
            st.subheader("多图空间稳定性")
            render_3d_visualization(bundle["results"])

    if method == "multisegment":
        st.info("多段式算法来自仓库实验方案，适合探索光照不均场景；现有基准没有人工标注真值。正式质量判定前，建议用已标注样本确认阈值标准和可接受误差。")

# Feedback and support
st.sidebar.divider()
with st.sidebar.expander("💬 使用反馈"):
    with st.form("feedback_form", clear_on_submit=True):
        feedback_type = st.selectbox("反馈类型", ["功能建议", "问题报告", "界面体验", "其他"])
        feedback_name = st.text_input("您的称呼（可选）")
        feedback_email = st.text_input("联系邮箱（可选）")
        feedback_message = st.text_area("反馈内容", placeholder="请描述使用场景、操作步骤和期望结果")
        feedback_images = st.file_uploader("上传截图（可选）", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="feedback_images")
        feedback_submitted = st.form_submit_button("发送反馈")
    if feedback_submitted:
        if len(feedback_message.strip()) < 5:
            st.warning("请填写至少 5 个字符的反馈内容。")
        else:
            sent, feedback_status = _record_feedback(feedback_type, feedback_name, feedback_email,
                                                      feedback_message.strip(), feedback_images)
            if sent:
                st.success(feedback_status)
            else:
                st.warning(feedback_status)
support_image = Path("img/赞赏码.jpg")
if support_image.exists():
    with st.sidebar.expander("💝 支持开发"):
        st.image(str(support_image), caption="感谢支持", use_column_width=True)
