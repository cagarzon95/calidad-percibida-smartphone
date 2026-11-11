"""Calcula las 68 características de cada imagen. El modelo usa 14 de ellas."""
import hashlib
import io

import cv2
import numpy as np
import scipy.stats
from PIL import Image, ImageCms, ImageOps
from skimage.feature import local_binary_pattern

SCENES = ["Animal", "Cityscape", "Human", "Indoor scene", "Landscape",
          "Night scene", "Plant", "Still-life", "Others"]
PREDICTORS = ["s512_luma_mean", "s512_luma_std", "heterogeneity_ratio", "luma_p95_minus_p05",
              "s512_gradient_mean", "s512_laplacian_var", "s512_fft_high_fraction", "s512_entropy",
              "hsv_s_mean", "lab_chroma_mean", "color_entropy", "neighbor_absdiff",
              "lbp_entropy", "tile_entropy_std"]

LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
SRGB = ImageCms.createProfile("sRGB")


def entropy(values):
    """Entropía en bits del histograma de valores entre 0 y 1."""
    counts = np.histogram(values, bins=256, range=(0, 1))[0]
    return float(scipy.stats.entropy(counts, base=2))


def color_entropy(rgb):
    """Entropía de los colores, con cada canal en 8 niveles."""
    levels = np.minimum((rgb * 8).astype(np.int32), 7)
    codes = levels[:, :, 0] * 64 + levels[:, :, 1] * 8 + levels[:, :, 2]
    return float(scipy.stats.entropy(np.bincount(codes.ravel(), minlength=512), base=2))


def tiles(values):
    """Divide la imagen en 16 bloques de 4 por 4."""
    return [block for row in np.array_split(values, 4, axis=0) for block in np.array_split(row, 4, axis=1)]


def spectrum(luma):
    """Fracción de energía de alta frecuencia y centroide del espectro."""
    height, width = luma.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    power = np.abs(np.fft.rfft2((luma - luma.mean()) * window, norm="ortho")) ** 2
    weights = np.full(power.shape[1], 2.0)
    weights[0] = 1
    if width % 2 == 0:
        weights[-1] = 1
    power *= weights[None, :]
    power[0, 0] = 0
    radius = np.hypot(np.fft.fftfreq(height)[:, None], np.fft.rfftfreq(width)[None, :])
    total = power.sum()
    if total <= 1e-20:
        return None, None
    return float(power[radius >= 0.25].sum() / total), float((power * radius).sum() / total)


def resize(image, scale):
    """Reduce la imagen para que su lado mayor mida como máximo `scale` píxeles."""
    width, height = image.size
    factor = min(1, scale / max(width, height))
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    return image.resize(size, Image.Resampling.LANCZOS)


def scale_features(image, scale):
    """Características de brillo, nitidez y detalle para un tamaño de imagen."""
    rgb = np.asarray(resize(image, scale), dtype=np.float32) / 255
    luma = rgb @ LUMA_WEIGHTS
    gradient = np.hypot(cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3) / 8,
                        cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3) / 8)
    tile_means = np.array([block.mean() for block in tiles(luma)])
    high, centroid = spectrum(luma)
    prefix = f"s{scale}_"
    features = {
        prefix + "luma_mean": float(luma.mean()),
        prefix + "luma_std": float(luma.std()),
        prefix + "gradient_mean": float(gradient.mean()),
        prefix + "laplacian_var": float(cv2.Laplacian(luma, cv2.CV_32F, ksize=1).var()),
        prefix + "fft_high_fraction": high,
        prefix + "tile_mean_std": float(tile_means.std()),
        prefix + "entropy": entropy(luma),
    }
    if scale == 512:
        features.update(full_size_features(rgb, luma, gradient, centroid))
    return features


def full_size_features(rgb, luma, gradient, centroid):
    """Características que solo se calculan con la imagen a 512 píxeles."""
    std = float(luma.std())
    skew = float(((luma - luma.mean()) ** 3).mean() / std ** 3) if std > 1e-8 else None
    gradient_means = np.array([block.mean() for block in tiles(gradient)])
    tile_luma = tiles(luma)
    tile_means = np.array([block.mean() for block in tile_luma])
    tile_entropies = np.array([entropy(block) for block in tile_luma])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
    mean_rgb = rgb.mean(axis=(0, 1))
    diffs = (np.abs(np.diff(luma, axis=0)).mean() + np.abs(np.diff(luma, axis=1)).mean()) / 2
    lbp = local_binary_pattern(np.round(luma * 255).astype(np.uint8), 8, 1, "uniform")[1:-1, 1:-1]
    counts = np.bincount(lbp.astype(int).ravel(), minlength=10)
    probs = counts[counts > 0] / counts.sum()
    gradient_mean = gradient_means.mean()

    features = {
        "analysis_width": luma.shape[1],
        "analysis_height": luma.shape[0],
        "luma_median": float(np.median(luma)),
        "luma_min": float(luma.min()),
        "luma_max": float(luma.max()),
        "luma_range": float(np.ptp(luma)),
        "luma_skew": skew,
        "gradient_p95": float(np.percentile(gradient, 95)),
        "tile_gradient_cv": float(gradient_means.std() / gradient_mean) if gradient_mean > 1e-8 else None,
        "tile_dark_fraction": float(np.mean(tile_means < 0.1)),
        "tile_bright_fraction": float(np.mean(tile_means > 0.9)),
        "tile_entropy_mean": float(tile_entropies.mean()),
        "tile_entropy_std": float(tile_entropies.std()),
        "fft_centroid": centroid,
        "hsv_s_mean": float(hsv[:, :, 1].mean()),
        "hsv_s_std": float(hsv[:, :, 1].std()),
        "hsv_v_mean": float(hsv[:, :, 2].mean()),
        "rgb_r_mean": float(mean_rgb[0]),
        "rgb_g_mean": float(mean_rgb[1]),
        "rgb_b_mean": float(mean_rgb[2]),
        "rgb_r_minus_g": float(mean_rgb[0] - mean_rgb[1]),
        "rgb_b_minus_g": float(mean_rgb[2] - mean_rgb[1]),
        "lab_l_mean": float(lab[:, :, 0].mean()),
        "lab_a_mean": float(lab[:, :, 1].mean()),
        "lab_b_mean": float(lab[:, :, 2].mean()),
        "lab_chroma_mean": float(chroma.mean()),
        "lab_chroma_std": float(chroma.std()),
        "tile_chroma_std": float(np.std([block.mean() for block in tiles(chroma)])),
        "color_entropy": color_entropy(rgb),
        "tile_color_entropy_std": float(np.std([color_entropy(block) for block in tiles(rgb)])),
        "neighbor_absdiff": float(diffs),
        "lbp_entropy": float(-(probs * np.log2(probs)).sum()),
    }
    levels = (1, 5, 25, 75, 95, 99)
    for level, value in zip(levels, np.percentile(luma, levels)):
        features[f"luma_p{level:02}"] = float(value)
    for level in (1, 5, 10):
        features[f"near_black_{level:02}"] = float(np.mean(luma <= level / 255))
    for level in (245, 250, 254):
        features[f"near_white_{level}"] = float(np.mean(luma >= level / 255))
        features[f"channel_clip_{level}"] = float(np.mean(rgb.max(axis=2) >= level / 255))
    return features


def extract_features(image):
    """Las 68 características de una imagen RGB, en tres tamaños."""
    result = {}
    for scale in (512, 256, 128):
        result.update(scale_features(image, scale))
    return result


def to_srgb(image, profile):
    """Pasa la imagen a sRGB si trae perfil de color. Si no se puede, la deja igual."""
    if not profile:
        return image
    try:
        return ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(profile)), SRGB, outputMode="RGB")
    except Exception:
        return image


def jpeg_bytes_to_features(content, image_id="upload"):
    """Características de un JPEG: orientación, sRGB, cálculo y huella digital."""
    with Image.open(io.BytesIO(content)) as raw:
        raw.load()
        image = to_srgb(ImageOps.exif_transpose(raw), raw.info.get("icc_profile"))
        values = extract_features(image.convert("RGB"))
    values["heterogeneity_ratio"] = (values["s512_tile_mean_std"] / values["s512_luma_std"]
                                     if values["s512_luma_std"] > 1e-8 else None)
    values["luma_p95_minus_p05"] = values["luma_p95"] - values["luma_p05"]
    return {"image_id": image_id, "sha256": hashlib.sha256(content).hexdigest(), **values}
