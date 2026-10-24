import rasterio
import numpy as np
from sklearn.model_selection import train_test_split


def create_train_test_masks(input_path, train_path, test_path, samples, test_size):
    # Load image
    with rasterio.open(input_path) as src:
        labels = src.read(1)
        profile = src.profile
    
    # Get classes
    unique_classes = np.unique(labels)
    # unique_classes = unique_classes[~np.isin(unique_classes, [0,2])]

    
    # Create masks
    train_mask = np.zeros_like(labels, dtype=np.uint8)
    test_mask = np.zeros_like(labels, dtype=np.uint8)
    
    np.random.seed(42)
    
    # Sample each class
    for cls in unique_classes:
        y, x = np.where(labels == cls)
        coords = np.column_stack([y, x])
        np.random.shuffle(coords)
        coords = coords[:samples]
        
        train_coords, test_coords = train_test_split(coords, test_size=test_size, random_state=42)
        
        train_mask[train_coords[:, 0], train_coords[:, 1]] = cls
        test_mask[test_coords[:, 0], test_coords[:, 1]] = cls
    
    # Save masks
    profile.update(dtype='uint8', count=1, compress='lzw')
    
    with rasterio.open(train_path, "w", **profile) as dst:
        dst.write(train_mask, 1)
    
    with rasterio.open(test_path, "w", **profile) as dst:
        dst.write(test_mask, 1)
    
    print(f"✅ Saved masks with {samples} samples per class across {len(unique_classes)} classes.")


def main():
    from projutils.config import config
    create_train_test_masks(
        config.new_final_class_labels,
        config.train_mask_path,
        config.test_mask_path,
        samples=config.samples,
        test_size=config.test_size
    )


if __name__ == "__main__":
    main()