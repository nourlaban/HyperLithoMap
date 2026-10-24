import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projutils.config import config
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, jaccard_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import rasterio
from tqdm import tqdm
import joblib


def load_tiff_input_and_labels(input_path, train_label_path, test_label_path):
    with rasterio.open(input_path) as src:
        input_data = src.read()
    input_data = np.transpose(input_data, (1, 2, 0))

    with rasterio.open(train_label_path) as src:
        TR = src.read(1)

    with rasterio.open(test_label_path) as src:
        TE = src.read(1)

    return input_data, TR, TE


def normalize_data(input_data):
    input_normalize = np.zeros(input_data.shape)
    for i in range(input_data.shape[2]):
        input_max = np.max(input_data[:, :, i])
        input_min = np.min(input_data[:, :, i])
        input_normalize[:, :, i] = (input_data[:, :, i] - input_min) / (input_max - input_min)
    return input_normalize


def prepare_train_test_data(input_normalize, TR, TE):
    train_mask = TR > 0
    test_mask = TE > 0

    x_train = input_normalize[train_mask]
    y_train = TR[train_mask]  

    x_test = input_normalize[test_mask]
    y_test = TE[test_mask]  

    return x_train, y_train, x_test, y_test


def evaluate_model(y_test, y_pred, classname_dict=None, save_dir=None):
    try:
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        print("======================================")

        IoU = jaccard_score(y_test, y_pred, average='macro')
        print(f"✅ IoU: {IoU:.4f}")
        print("======================================")

        # Get unique labels and their names
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        
        label_dict = {v: f"{k} ({v})" for k, v in classname_dict.items()} # type: ignore
        classname_dict = label_dict
        target_names = [classname_dict.get(label, str(label)) for label in unique_labels]

        # Classification report with class names
        report = classification_report(y_test, y_pred, 
                                       labels=unique_labels,
                                       target_names=target_names, 
                                       digits=4)
        print("\n📊 Classification Report:")
        print(report)

        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Use class names for confusion matrix labels
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Normalized Confusion Matrix")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_dir:
            eval_path = os.path.join(save_dir, 'xgb_model_evaluation')
            os.makedirs(eval_path, exist_ok=True)
            plt.savefig(os.path.join(eval_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved")

            # Write results to text file
            results_path = os.path.join(eval_path, 'evaluation_results.txt')
            with open(results_path, 'w') as f:
                f.write("===== Model Evaluation Results =====\n\n")
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"IoU (Jaccard Score): {IoU:.4f}\n\n")
                f.write("Class Mapping:\n")
                for label in unique_labels:
                    name = classname_dict.get(label, str(label)) 
                    f.write(f"  {label}: {name}\n")
                f.write("\nClassification Report:\n")
                f.write(report) # type: ignore
                f.write("\n\nConfusion Matrix:\n")
                f.write(str(cm))
            print(f"✅ Results saved to: {results_path}")
        
        plt.close()

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        plt.close()


def predict_full_raster(xgb_model, input_data_raw, height, width):
    y_pred_full = []
    for i in tqdm(range(0, len(input_data_raw), 1000), desc="Predicting"):
        chunk = input_data_raw[i:i + 1000]
        y_pred_chunk = xgb_model.predict(chunk)
        y_pred_full.extend(y_pred_chunk)

    preprocessed_mask = np.array(y_pred_full).reshape(height, width)
    return preprocessed_mask


def save_output_raster(preprocessed_mask, input_path, output_path):
    with rasterio.open(input_path) as src:
        profile = src.profile

    profile.update(
        dtype=rasterio.int16,
        count=1,
        compress='lzw'
    )

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(preprocessed_mask.astype(rasterio.int16), 1)

    print(f"✅ Saved prediction map to: {output_path}")


def train_model(all_classes,x_train, y_train, x_test, y_test, input_data_raw, height, width):
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Training
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(x_train, y_train_enc)

    # Predictions on test set
    y_pred_enc = xgb_model.predict(x_test)
    y_pred = le.inverse_transform(y_pred_enc)

    # Evaluation
    model_dir = config.save_model_path
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"✅ Created directory: {model_dir}")

    evaluate_model(y_test, y_pred, classname_dict=all_classes, save_dir=model_dir)

    # Predict full raster
    preprocessed_mask = predict_full_raster(xgb_model, input_data_raw, height, width)

    # Save output raster
    save_output_raster(preprocessed_mask + 1 , config.tifpath, config.output_xgb_tif_path)

    # Save trained XGBoost model
    
    model_path = os.path.join(model_dir, 'xgb_model.pkl')
    joblib.dump(xgb_model, model_path)
    print(f"✅ Model saved to: {model_path}")

    # Save label encoder
    le_path = os.path.join(config.save_model_path, 'xgb_label_encoder.pkl')
    joblib.dump(le, le_path)
    print(f"✅ Label encoder saved to: {le_path}")


def test_model(input_data_raw, height, width):
    # Load model
    model_path = os.path.join(config.save_model_path, 'xgb_model.pkl')
    xgb_model = joblib.load(model_path)
    print(f"✅ Loaded model from: {model_path}")

    # Predict full raster
    preprocessed_mask = predict_full_raster(xgb_model, input_data_raw, height, width)

    # Save output raster
    save_output_raster(preprocessed_mask, config.tifpath, config.output_xgb_tif_path)


def main(all_classes):
    # Arguments
    dataset = 'aborashed'
    flag_test = 'train'  # 'train' or 'test'

    # Load data
    if dataset == 'aborashed':
        input_data, TR, TE = load_tiff_input_and_labels(
            config.tifpath,
            config.train_mask_path,
            config.test_mask_path
        )
    else:
        raise ValueError("Unknown dataset")

    # Load label image and its color map
    with rasterio.open(config.label_tif_path) as color_mat:
        label_data = color_mat.read(1)
        cmap = color_mat.colormap(1)

    # Get dimensions
    height, width, band = input_data.shape
    print(f"height={height}, width={width}, band={band}")

    # Normalize data
    input_normalize = normalize_data(input_data)
    input_data_raw = input_normalize.reshape(-1, band)

    # Prepare training and testing data
    x_train, y_train, x_test, y_test = prepare_train_test_data(input_normalize, TR, TE)
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")

    # Train or test
    if flag_test == "train":
        train_model(all_classes,x_train, y_train, x_test, y_test, input_data_raw, height, width)
    else:
        test_model(input_data_raw, height, width)


