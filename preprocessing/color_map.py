import rasterio
import os


_DEFAULT_COLORMAP = {
    0: (0, 0, 0), 1: (127, 127, 127), 2: (255, 255, 255),
    3: (255, 0, 0), 4: (0, 255, 0), 5: (0, 0, 255),
    6: (255, 255, 0), 7: (255, 0, 255), 8: (0, 255, 255),
    9: (160, 32, 240), 10: (255, 165, 0), 11: (128, 0, 0),
    12: (0, 128, 128)
}


def _load_clr(clr_path):
    """Parse a .clr file (format: value R G B Alpha value) into a colormap dict."""
    colormap = {0: (0, 0, 0)}
    with open(clr_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                value, r, g, b = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                colormap[value] = (r, g, b)
    return colormap


def apply_colormap_to_tiff(input_path, output_path, clr_path=None):
    """Apply colormap (from a .clr file, else a default palette) to a classified TIFF."""

    if clr_path and os.path.exists(clr_path):
        colormap = _load_clr(clr_path)
    else:
        colormap = _DEFAULT_COLORMAP

    with rasterio.open(input_path) as src:
        label_data = src.read(1)
        profile = src.profile.copy()

    profile.update({'count': 1, 'dtype': 'uint8', 'compress': 'lzw'})

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(label_data, 1)
        dst.write_colormap(1, colormap)

    print(f"✅ Colorized TIFF saved to: {output_path}")


# Usage
# apply_colormap_to_tiff(config.new_final_class_labels, config.label_tif_path)
