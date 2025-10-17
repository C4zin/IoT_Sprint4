import os
import cv2
import time
import sys
import glob
import pickle
import shutil
import importlib
import subprocess
import numpy as np
import streamlit as st

# ============================================================
# 🔧 Instalação/garantia de dependências (fallback)
#  - Em plataformas gerenciadas (Streamlit Cloud/Render), prefira requirements.txt
#  - Isso aqui só dá uma ajudinha quando roda local/Colab
# ============================================================
def ensure(pkg, pip_name=None):
    pip_name = pip_name or pkg
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

# essenciais
ensure("ultralytics")
ensure("supervision", "supervision==0.21.0")
ensure("cv2", "opencv-python-headless==4.10.0.84")
ensure("lapx", "lapx>=0.5.9")

# tentar streamlit-webrtc (opcional)
try:
    ensure("streamlit_webrtc")
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

# ============================================================
# Imports principais do pipeline
# ============================================================
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import supervision as sv

# ============================================================
# Ajustes de cache p/ evitar arquivos corrompidos
# ============================================================
# Use um cache local controlado pelo app (evita colisões em ambientes gerenciados)
os.environ.setdefault("ULTRALYTICS_CACHE_DIR", ".ultra_cache")
try:
    os.makedirs(os.environ["ULTRALYTICS_CACHE_DIR"], exist_ok=True)
except Exception:
    pass

def _clean_ultralytics_cache_for(weights_name: str):
    """
    Remove arquivos possivelmente corrompidos do cache do Ultralytics/torch
    relacionados ao 'weights_name' (ex.: 'yolov8n.pt').
    """
    try:
        candidates = []
        stem = os.path.splitext(os.path.basename(weights_name))[0]  # 'yolov8n' de 'yolov8n.pt'

        # 1) Pasta padrão de pesos do Ultralytics (SETTINGS)
        weights_dir = SETTINGS.get("weights_dir", None)
        if weights_dir and os.path.isdir(weights_dir):
            candidates += glob.glob(os.path.join(weights_dir, f"{stem}*"))

        # 2) Nosso cache dedicado
        ultra_cache = os.environ.get("ULTRALYTICS_CACHE_DIR")
        if ultra_cache and os.path.isdir(ultra_cache):
            candidates += glob.glob(os.path.join(ultra_cache, "*"))

        # 3) Cache do torch (às vezes armazena o download bruto)
        torch_home = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))
        if torch_home and os.path.isdir(torch_home):
            candidates += glob.glob(os.path.join(torch_home, "**", f"*{stem}*"), recursive=True)

        # Remove arquivos/diretórios candidatos
        for p in set(candidates):
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    os.remove(p)
            except Exception:
                # não deixa a limpeza quebrar a execução
                pass
    except Exception:
        pass


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="MotoTrack Vision - Streamlit", layout="wide")
st.title("🏍️ MotoTrack Vision — YOLOv8 + ByteTrack (Streamlit)")
st.caption("Detecção + Rastreamento de **motos** em vídeo, com métricas em tempo (quase) real, exportação CSV e MP4.")

# -----------------------------
# Abas principais
# -----------------------------
tab_sys, tab_app = st.tabs(["🖥️ Sistema", "🎬 Processamento"])

with tab_sys:
    st.subheader("Informações do Sistema/GPU")
    # Mostrar info de versões
    import platform
    st.write({
        "python": platform.python_version(),
        "numpy": np.__version__ if 'np' in globals() else None,
        "opencv": cv2.__version__ if 'cv2' in globals() else None,
        "ultralytics": __import__("ultralytics").__version__,
    })
    # Tentar mostrar nvidia-smi se existir
    import shutil as _shutil, subprocess as _subprocess
    if _shutil.which("nvidia-smi"):
        try:
            out = _subprocess.check_output(["nvidia-smi"], text=True)
            st.code(out)
        except Exception as e:
            st.warning(f"Falha ao executar nvidia-smi: {e}")
    else:
        st.info("GPU NVIDIA não detectada (ou `nvidia-smi` indisponível).")

with tab_app:
    # -----------------------------
    # Controles (sidebar)
    # -----------------------------
    with st.sidebar:
        st.header("Parâmetros")
        model_name = st.selectbox("Modelo YOLO", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
        conf = st.slider("Confiança mínima", 0.1, 0.9, 0.35, 0.05)
        iou = st.slider("IoU NMS", 0.1, 0.9, 0.5, 0.05)
        max_frames = st.number_input("Limite de frames (0 = todos)", min_value=0, value=0, step=1)
        save_output = st.checkbox("Salvar resultado em MP4", value=True)
        output_path = st.text_input("Caminho do MP4 de saída", value="video_output.mp4")
        tracker_cfg = st.text_input("Arquivo de tracker (.yaml)", value="bytetrack.yaml")

        st.markdown("---")
        if st.button("♻️ Limpar estado"):
            for k in list(st.session_state.keys()):
                try:
                    del st.session_state[k]
                except Exception:
                    pass
            try:
                st.cache_resource.clear()
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

    run_button = st.button("▶️ Processar vídeo enviado")

    # -----------------------------
    # Carregar modelo (cache com proteção contra .pt corrompido)
    # -----------------------------
    @st.cache_resource(show_spinner=True)
    def load_model_safely(name: str):
        """
        Tenta carregar o YOLO. Se falhar por UnpicklingError/RuntimeError/ValueError,
        limpa caches relevantes e re-tenta uma vez.
        """
        try:
            return YOLO(name)
        except (pickle.UnpicklingError, RuntimeError, ValueError) as e:
            # Limpa coisas possivelmente quebradas e tenta novamente
            _clean_ultralytics_cache_for(name)
            return YOLO(name)

    model = load_model_safely(model_name)

    # -----------------------------
    # Upload de vídeo
    # -----------------------------
    uploaded = st.file_uploader(
        "Envie um arquivo de vídeo (mp4, avi, mov, mkv…)", type=None, accept_multiple_files=False
    )

    # Layout principal
    video_col, metrics_col = st.columns([3, 1])
    frame_placeholder = video_col.empty()
    track_table_placeholder = metrics_col.empty()
    csv_preview_placeholder = st.empty()

    # -----------------------------
    # Utilitários
    # -----------------------------
    def _save_to_temp(uploaded_file) -> str:
        suffix = os.path.splitext(uploaded_file.name)[-1] or ".mp4"
        # mantém entrada separada por nome (evita sobrescrever)
        base = os.path.splitext(os.path.basename(uploaded_file.name))[0] or "input"
        temp_path = os.path.join(f"data_cache_input_{base}{suffix}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        return temp_path

    # -----------------------------
    # Pipeline de processamento
    # -----------------------------
    def process_video(input_path: str):
        writer = None
        unique_ids = set()
        frame_count = 0
        start_time = time.time()
        logs = []

        stream = model.track(
            source=input_path,
            stream=True,
            conf=conf,
            iou=iou,
            classes=[3],  # COCO: motorcycle
            tracker=tracker_cfg,
            persist=True,
            verbose=False,
        )

        for result in stream:
            frame = result.plot()
            frame_count += 1

            ids = []
            if result.boxes is not None and result.boxes.id is not None:
                try:
                    ids = result.boxes.id.cpu().numpy().tolist()
                except Exception:
                    # fallback seguro se vier tensor vazio/None inesperado
                    ids = []
                for tid in ids:
                    unique_ids.add(int(tid))

            logs.append({
                "frame": frame_count,
                "ids_no_frame": ids,
                "motos_no_frame": len(ids),
                "motos_unicas": len(unique_ids),
            })

            if save_output and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, 24.0, (w, h))

            if save_output and writer is not None:
                writer.write(frame)

            frame_placeholder.image(frame[:, :, ::-1], channels="RGB")

            elapsed = max(time.time() - start_time, 1e-6)
            fps = frame_count / elapsed
            metrics_col.metric("FPS (estimado)", f"{fps:0.1f}")
            metrics_col.metric("Motos no frame", str(len(ids)))
            metrics_col.metric("Motos únicas", str(len(unique_ids)))

            if len(unique_ids) > 0:
                # mostra amostra dos últimos 10 IDs
                try:
                    sample_ids = sorted(unique_ids)[-10:]
                except Exception:
                    sample_ids = list(unique_ids)[:10]
                track_table_placeholder.write({"IDs rastreadas (amostra)": sample_ids})

            if max_frames and frame_count >= max_frames:
                break

        if writer is not None:
            writer.release()

        return {
            "frames": frame_count,
            "unique_ids": len(unique_ids),
            "out": output_path if save_output else None,
            "logs": logs,
        }

    # -----------------------------
    # Execução
    # -----------------------------
    summary = None

    if run_button:
        if uploaded is None:
            st.error("Envie um vídeo primeiro.")
        else:
            in_path = _save_to_temp(uploaded)
            with st.spinner("Processando vídeo..."):
                summary = process_video(in_path)

            st.success(f"Concluído: {summary['frames']} frames • {summary['unique_ids']} motos únicas.")

            if save_output and summary.get("out"):
                st.video(summary["out"])

            if summary.get("logs"):
                import pandas as pd
                df_logs = pd.DataFrame(summary["logs"])
                csv_preview_placeholder.dataframe(df_logs.head(50), use_container_width=True)
                csv_bytes = df_logs.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Baixar CSV com métricas",
                    data=csv_bytes,
                    file_name="rastreamento_motos.csv",
                    mime="text/csv",
                )
