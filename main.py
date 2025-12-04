import os
import sys
import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import streamlit as st

try:
    import torch
    import numpy as np
    from PIL import Image
except Exception as e:
    TORCH_IMPORT_ERROR = e
else:
    TORCH_IMPORT_ERROR = None


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def auto_device(env_device: Optional[str] = None) -> str:
    if env_device:
        return env_device
    if TORCH_IMPORT_ERROR is not None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_sd_pipeline(model_name: str, device: str = "cuda"):
    """
    Loads Stable Diffusion pipeline once and caches it for the Streamlit session.
    """
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        safety_checker=None,
        revision="fp16" if dtype == torch.float16 else None,
    )
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception:
        pass
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

def generate_images(prompt: str,
                    out_dir: str = "outputs",
                    n_samples: int = 3,
                    seed: Optional[int] = None,
                    guidance_scale: float = 7.5,
                    steps: int = 25,
                    model_name: Optional[str] = None,
                    device: Optional[str] = None) -> List[str]:
    """
    Generate images using Stable Diffusion pipeline loaded via load_sd_pipeline.
    Returns list of saved file paths.
    """
    if TORCH_IMPORT_ERROR is not None:
        raise TORCH_IMPORT_ERROR

    model_name = model_name or os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
    device = auto_device(device or os.getenv("DEVICE"))
    pipe = load_sd_pipeline(model_name, device=device)
    ensure_dir(out_dir)

    generator = None
    if seed is not None:
        if device.startswith("cuda"):
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)

    out_paths = []
    tbase = int(time.time() * 1000)
    for i in range(n_samples):
        out = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=steps, generator=generator)
        img = out.images[0]
        filename = f"sample_{tbase}_{i+1}.png"
        path = Path(out_dir) / filename
        img.save(path)
        out_paths.append(str(path))
    return out_paths

@st.cache_resource
def load_clip(model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def encode_text(model: CLIPModel, processor: CLIPProcessor, text: str, device: str = "cpu"):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def encode_image(model: CLIPModel, processor: CLIPProcessor, image_path: str, device: str = "cpu"):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

DEFAULT_REALISM_DESCRIPTORS = [
    "a photograph",
    "a high quality photograph",
    "a realistic photo",
    "a professional photograph",
]

def realism_score_via_descriptors(clip_model, processor, image_path: str, device: str = "cpu", descriptors: Optional[List[str]] = None) -> float:
    descriptors = descriptors or DEFAULT_REALISM_DESCRIPTORS
    img_emb = encode_image(clip_model, processor, image_path, device)
    sims = []
    for d in descriptors:
        t_emb = encode_text(clip_model, processor, d, device)
        sims.append(cosine_sim(img_emb, t_emb))
    mean_sim = float(np.mean(sims))
    return (mean_sim + 1.0) / 2.0

def final_score(alignment: float, realism: Optional[float], alignment_weight: float = 0.7) -> float:
    if realism is None:
        return float(alignment)
    return float(alignment_weight * alignment + (1.0 - alignment_weight) * realism)

def evaluate_images(images_dir: str,
                    prompt: str,
                    clip_model_name: Optional[str] = None,
                    device: Optional[str] = None,
                    compute_realism: bool = True,
                    alignment_weight: float = 0.7,
                    auto_generate_if_empty: bool = False,
                    gen_n_samples: int = 3,
                    gen_seed: Optional[int] = None,
                    gen_guidance: float = 7.5,
                    gen_steps: int = 25,
                    sd_model_name: Optional[str] = None) -> List[Dict[str, Any]]:

    if TORCH_IMPORT_ERROR is not None:
        raise TORCH_IMPORT_ERROR

    images_dir = Path(images_dir)
    ensure_dir(images_dir.as_posix())

    pngs = sorted([p for p in images_dir.glob("*.png")])
    if len(pngs) == 0 and auto_generate_if_empty:
        generate_images(prompt, out_dir=images_dir.as_posix(), n_samples=gen_n_samples,
                        seed=gen_seed, guidance_scale=gen_guidance, steps=gen_steps,
                        model_name=sd_model_name, device=device)
        pngs = sorted([p for p in images_dir.glob("*.png")])

    if len(pngs) == 0:
        raise FileNotFoundError(f"No PNG images found in: {images_dir}")

    clip_model_name = clip_model_name or os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    device = auto_device(device or os.getenv("DEVICE"))
    clip_model, processor = load_clip(clip_model_name, device=device)
    text_emb = encode_text(clip_model, processor, prompt, device)

    results = []
    for p in pngs:
        img_emb = encode_image(clip_model, processor, str(p), device)
        sim = cosine_sim(text_emb, img_emb)
        alignment = (sim + 1.0) / 2.0
        realism = None
        if compute_realism:
            try:
                realism = realism_score_via_descriptors(clip_model, processor, str(p), device)
            except Exception:
                realism = None
        fscore = final_score(alignment, realism, alignment_weight=alignment_weight)
        results.append({
            "image": str(p),
            "clip_similarity": float(sim),
            "alignment_score": float(alignment),
            "realism_score": (None if realism is None else float(realism)),
            "final_score": float(fscore),
        })

    return sorted(results, key=lambda r: r["final_score"], reverse=True)

def plot_scores(results: List[Dict[str, Any]], out_path: str = "outputs/scores_bar.png"):
    ensure_dir(Path(out_path).parent.as_posix())
    labels = [Path(r["image"]).name for r in results]
    scores = [r["final_score"] for r in results]
    plt.figure(figsize=(max(6, len(labels)*1.2), 4))
    plt.bar(range(len(scores)), scores, tick_label=labels)
    plt.ylim(0,1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Final score (0-1)")
    plt.title("Image final scores (higher is better)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def run_streamlit_ui():
    st.set_page_config(layout="wide", page_title="Image Generation Evaluator")
    st.title("Image Generation Evaluator")

    if TORCH_IMPORT_ERROR is not None:
        st.error(f"Missing or broken ML dependencies: {TORCH_IMPORT_ERROR}")
        st.stop()

    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = []

    with st.sidebar:
        st.header("Generation / Evaluation settings")
        sd_model = st.text_input("Stable Diffusion model", value=os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5"))
        clip_model = st.text_input("CLIP model", value=os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32"))
        device_option = st.selectbox("Device", ["auto","cuda","cpu"], index=0)
        outputs_dir = st.text_input("Outputs directory", value="outputs")
        ensure_dir(outputs_dir)

    prompt = st.text_area("Prompt", value="A futuristic city with neon lights in Cyberpunk style.", height=120)
    c1, c2 = st.columns([1,1])

    with c1:
        n_samples = st.number_input("Number of images", min_value=1, max_value=10, value=3)
        seed = st.number_input("Seed (0 for random)", min_value=0, value=0)
        guidance = st.slider("Guidance scale", 1.0, 20.0, 7.5)
        steps = st.slider("Steps", 10, 50, 25)

    with c2:
        compute_realism = st.checkbox("Compute Realism Score", value=True)
        alignment_weight = st.slider("Alignment Weight", 0.0, 1.0, 0.7)
        auto_generate_if_empty = st.checkbox("Auto-generate if no images", value=False)

    st.markdown("---")
    gen_col, eval_col = st.columns(2)

    with gen_col:
        if st.button("Generate Images"):
            s = None if int(seed) == 0 else int(seed)
            try:
                with st.spinner("Generating images (this may take a while on first run)..."):
                    device = None if device_option == "auto" else device_option
                    imgs = generate_images(prompt, out_dir=outputs_dir, n_samples=int(n_samples), seed=s,
                                           guidance_scale=float(guidance), steps=int(steps),
                                           model_name=sd_model, device=device)
                    st.session_state.generated_images = imgs
                st.success(f"Generated {len(imgs)} images.")
            except Exception as e:
                st.exception(e)

    with eval_col:
        if st.button("Evaluate Images"):
            try:
                with st.spinner("Evaluating images..."):
                    device = None if device_option == "auto" else device_option
                    results = evaluate_images(outputs_dir, prompt,
                                              clip_model_name=clip_model,
                                              device=device,
                                              compute_realism=compute_realism,
                                              alignment_weight=float(alignment_weight),
                                              auto_generate_if_empty=auto_generate_if_empty,
                                              gen_n_samples=int(n_samples),
                                              gen_seed=(None if int(seed) == 0 else int(seed)),
                                              gen_guidance=float(guidance),
                                              gen_steps=int(steps),
                                              sd_model_name=sd_model)
                    st.session_state.evaluation_results = results
                    save_json(results, Path(outputs_dir) / "evaluation_results.json")
                    chart_path = plot_scores(results, out_path=Path(outputs_dir) / "scores_bar.png")
                st.success("Evaluation complete.")
            except Exception as e:
                st.exception(e)

    st.markdown("---")
    st.header("Generated Images")
    if st.session_state.generated_images:
        cols = st.columns(min(4, len(st.session_state.generated_images)))
        for i, img in enumerate(st.session_state.generated_images):
            with cols[i % 4]:
                st.image(img, caption=Path(img).name, use_column_width=True)
    else:
        st.info("No generated images in session. Click 'Generate Images' to create them.")

    st.markdown("---")
    st.header("Evaluation Results")
    if st.session_state.evaluation_results:
        top = st.session_state.evaluation_results[:8]
        cols = st.columns(min(4, len(top)))
        for i, r in enumerate(top):
            with cols[i % 4]:
                realism_text = "NA" if r["realism_score"] is None else f"{r['realism_score']:.3f}"
                caption = f"Final: {r['final_score']:.3f} | Align: {r['alignment_score']:.3f} | Realism: {realism_text}"
                st.image(r["image"], caption=caption, use_column_width=True)
        st.subheader("Full JSON results")
        st.json(st.session_state.evaluation_results)
        scores_png = Path(outputs_dir) / "scores_bar.png"
        if scores_png.exists():
            st.image(str(scores_png), caption="Score Visualization")
    else:
        st.info("No evaluation results yet. Click 'Evaluate Images' to run evaluation.")

if __name__ == "__main__":
    is_streamlit_env = (
        any("streamlit" in a for a in sys.argv) or
        ("STREAMLIT_SERVER_PORT" in os.environ) or
        ("STREAMLIT_RUN_MAIN" in os.environ)
    )
    if is_streamlit_env:
        run_streamlit_ui()
    else:
        run_streamlit_ui()

# streamlit run main.py in terminal