import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import numpy as np
import rasterio
import argparse
import os
import sys
import time
from sklearn.metrics import confusion_matrix,accuracy_score, jaccard_score, classification_report
from hsmodel.vit_pytorch import ViT
import matplotlib.pyplot as plt
import seaborn as sns


# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projutils.config import config

def parse_arguments():
    """Parse command line arguments."""
    class Args:
        dataset = 'aborashed'          # choices: ['aborashed']
        flag_train = True
        flag_test = True        
        flag_predict = True
        mode = 'ViT'                   # choices: ['ViT', 'CAF']
        gpu_id = '0'
        seed = 0
        batch_size = 128
        test_freq = 5
        patches = 3
        band_patches = 3
        epochs = 200
        learning_rate = 5e-4
        gamma = 0.9
        weight_decay = 0
    
    return Args()

def load_tiff_data(input_path, train_label_path, test_label_path):
    """Load TIFF input data and labels."""
    with rasterio.open(input_path) as src:
        input_data = np.transpose(src.read(), (1, 2, 0))
    
    with rasterio.open(train_label_path) as src:
        train_labels = src.read(1)
    
    with rasterio.open(test_label_path) as src:
        test_labels = src.read(1)
    
    return input_data, train_labels, test_labels

def normalize_input(input_data, nodata=-9999.0):
    """Normalize input data by band, excluding nodata pixels from min/max."""
    input_normalize = np.zeros_like(input_data, dtype=float)
    for i in range(input_data.shape[2]):
        band = input_data[:, :, i]
        valid = band[band != nodata]
        if len(valid) > 0 and valid.max() > valid.min():
            input_normalize[:, :, i] = np.where(
                band != nodata,
                (band - valid.min()) / (valid.max() - valid.min()),
                0.0
            )
    return input_normalize

def choose_train_test_points(train_data, test_data, true_data, num_classes):
    """Select train, test, and true points for each class."""
    pos_train, pos_test, pos_true = {}, {}, {}
    number_train, number_test, number_true = [], [], []
    
    for i in range(num_classes):
        pos_train[i] = np.argwhere(train_data == (i + 1))
        pos_test[i] = np.argwhere(test_data == (i + 1))
        number_train.append(pos_train[i].shape[0])
        number_test.append(pos_test[i].shape[0])
    
    for i in range(num_classes + 1):
        pos_true[i] = np.argwhere(true_data == i)
        number_true.append(pos_true[i].shape[0])
    
    total_pos_train = np.concatenate([pos_train[i] for i in range(num_classes)], axis=0).astype(int)
    total_pos_test = np.concatenate([pos_test[i] for i in range(num_classes)], axis=0).astype(int)
    total_pos_true = np.concatenate([pos_true[i] for i in range(num_classes + 1)], axis=0).astype(int)
    
    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true

def mirror_hsi(height, width, band, input_normalize, patch=5):
    """Apply mirror padding to hyperspectral image."""
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    
    # Center region
    mirror_hsi[padding:padding + height, padding:padding + width, :] = input_normalize
    
    # Mirror edges
    for i in range(padding):
        mirror_hsi[padding:height + padding, i, :] = input_normalize[:, padding - i - 1, :]
        mirror_hsi[padding:height + padding, width + padding + i, :] = input_normalize[:, width - 1 - i, :]
        mirror_hsi[i, :, :] = mirror_hsi[2 * padding - i - 1, :, :]
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]
    
    print(f"Patch size: {patch}")
    print(f"Mirror image shape: {mirror_hsi.shape}")
    return mirror_hsi

def gain_neighborhood_pixel(mirror_image, point, idx, patch=5):
    """Extract patch around a given point."""
    x, y = point[idx]
    return mirror_image[x:x + patch, y:y + patch, :]

def gain_neighborhood_band(x_data, band, band_patch, patch=5):
    """Extract neighboring bands for patch data."""
    nn = band_patch // 2
    x_reshape = x_data.reshape(x_data.shape[0], patch * patch, band)
    x_band = np.zeros((x_data.shape[0], patch * patch * band_patch, band), dtype=float)
    
    x_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_reshape
    
    for i in range(nn):
        x_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_reshape[:, :, band - i - 1:]
        x_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_reshape[:, :, :band - i - 1]
        x_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_reshape[:, :, i + 1:]
        x_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_reshape[:, :, :i + 1]
    
    return x_band

def prepare_data(mirror_image, band, train_points, test_points, patch=5, band_patch=3):
    """Prepare training and testing data patches."""
    x_train = np.zeros((train_points.shape[0], patch, patch, band), dtype=np.float32)
    x_test = np.zeros((test_points.shape[0], patch, patch, band), dtype=np.float32)

    for i in range(train_points.shape[0]):
        x_train[i] = gain_neighborhood_pixel(mirror_image, train_points, i, patch)
    for i in range(test_points.shape[0]):
        x_test[i] = gain_neighborhood_pixel(mirror_image, test_points, i, patch)

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)

    print(f"x_train_band shape: {x_train_band.shape}, type: {x_train_band.dtype}")
    print(f"x_test_band shape: {x_test_band.shape}, type: {x_test_band.dtype}")

    return x_train_band, x_test_band


def predict_full_image_chunked(model, mirror_image, total_pos_true, band, patch, band_patch, chunk_size=512):
    """Predict class for all pixels in memory-safe chunks (avoids OOM on large images)."""
    model.eval()
    n = total_pos_true.shape[0]
    predictions = np.zeros(n, dtype=np.int32)

    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_pts = total_pos_true[start:end]
            sz = end - start

            x_chunk = np.zeros((sz, patch, patch, band), dtype=np.float32)
            for i in range(sz):
                x, y = chunk_pts[i]
                x_chunk[i] = mirror_image[x:x + patch, y:y + patch, :]

            x_band = gain_neighborhood_band(x_chunk, band, band_patch, patch)
            x_tensor = torch.from_numpy(x_band.transpose(0, 2, 1)).float().cuda()
            preds = model(x_tensor).argmax(dim=1).cpu().numpy()
            predictions[start:end] = preds

    return predictions

def create_labels(number_train, number_test, number_true, num_classes):
    """Create labels for training, testing, and true data."""
    y_train = np.concatenate([np.full(number_train[i], i) for i in range(num_classes)])
    y_test = np.concatenate([np.full(number_test[i], i) for i in range(num_classes)])
    y_true = np.concatenate([np.full(number_true[i], i) for i in range(num_classes + 1)])
    
    print(f"y_train shape: {y_train.shape}, type: {y_train.dtype}")
    print(f"y_test shape: {y_test.shape}, type: {y_test.dtype}")
    print(f"y_true shape: {y_true.shape}, type: {y_true.dtype}")
    
    return y_train, y_test, y_true

class AverageMeter:
    """Track average metrics during training/validation."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.avg = self.sum = self.cnt = 0
    
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    """Compute accuracy for given top-k values."""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = [correct[:k].view(-1).float().sum(0).mul_(100.0 / batch_size) for k in topk]
    return res, target, pred.squeeze()

def train_epoch(model, train_loader, criterion, optimizer):
    """Train model for one epoch."""
    objs = AverageMeter()
    top1 = AverageMeter()
    tar, pre = np.array([]), np.array([])
    
    model.train()
    for batch_data, batch_target in train_loader:
        batch_data, batch_target = batch_data.cuda(), batch_target.cuda()
        
        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        
        prec1, t, p = accuracy(batch_pred, batch_target)
        objs.update(loss.item(), batch_data.shape[0])
        top1.update(prec1[0].item(), batch_data.shape[0])
        tar = np.append(tar, t.cpu().numpy())
        pre = np.append(pre, p.cpu().numpy())
    
    return top1.avg, objs.avg, tar, pre

def valid_epoch(model, valid_loader, criterion):
    """Validate model for one epoch."""
    objs = AverageMeter()
    top1 = AverageMeter()
    tar, pre = np.array([]), np.array([])
    
    model.eval()
    with torch.no_grad():
        for batch_data, batch_target in valid_loader:
            batch_data, batch_target = batch_data.cuda(), batch_target.cuda()
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
            
            prec1, t, p = accuracy(batch_pred, batch_target)
            objs.update(loss.item(), batch_data.shape[0])
            top1.update(prec1[0].item(), batch_data.shape[0])
            tar = np.append(tar, t.cpu().numpy())
            pre = np.append(pre, p.cpu().numpy())
    
    return tar, pre

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
            eval_path = os.path.join(save_dir, 'VITransformer_model_evaluation')
            os.makedirs(eval_path, exist_ok=True)
            plt.savefig(os.path.join(eval_path, 'VITransformer_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved")

            # Write results to text file
            results_path = os.path.join(eval_path, 'VITransformer_evaluation_results.txt')
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

def test_epoch(model, test_loader):
    """Test model and return predictions."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch_data, _ in test_loader:
            batch_data = batch_data.cuda()
            outputs = model(batch_data)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
    
    return torch.cat(all_preds).numpy()

def calculate_metrics(tar, pre):
    """Calculate classification metrics from confusion matrix."""
    accuracy = accuracy_score(tar, pre)
    matrix = confusion_matrix(tar, pre)
    number = np.sum(np.diag(matrix))
    total = np.sum(matrix)
    AA = np.diag(matrix) / np.sum(matrix, axis=1)
    OA = number / total
    AA_mean = np.mean(AA)
    pe = np.sum(np.sum(matrix, axis=0) * np.sum(matrix, axis=1)) / (total ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA, accuracy

def save_prediction_map(prediction_matrix, height, width, config):
    """Save prediction matrix as GeoTIFF."""
    with rasterio.open(config.tifpath) as src:
        profile = src.profile
    
    profile.update(dtype=rasterio.int16, count=1, compress='lzw')
    
    with rasterio.open(config.outputVIT_tif_path, 'w', **profile) as dst:
        dst.write(prediction_matrix.astype(rasterio.int16), 1)
    
    print(f"✅ Saved prediction map to: {config.outputVIT_tif_path}")

def main(all_classes):
    """Main function for training and testing the hyperspectral classifier."""
    args = parse_arguments()
    if getattr(config, 'epochs', None):
        args.epochs = config.epochs

    # Set random seeds and CUDA settings
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Load and preprocess data
    if args.dataset != 'aborashed':
        raise ValueError("Unknown dataset")
    
    input_data, train_labels, test_labels = load_tiff_data(
        config.tifpath, config.train_mask_path, config.test_mask_path
    )
    
    with rasterio.open(config.label_tif_path) as color_mat:
        label_data = color_mat.read(1)
    
    input_normalize = normalize_input(input_data)
    height, width, band = input_data.shape
    num_classes = np.max(train_labels)
    print(f"Input data: height={height}, width={width}, bands={band}")
    
    # Prepare data points and patches
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = \
        choose_train_test_points(train_labels, test_labels, train_labels + test_labels, num_classes)
    
    mirror_image = mirror_hsi(height, width, band, input_normalize, args.patches)
    x_train_band, x_test_band = prepare_data(
        mirror_image, band, total_pos_train, total_pos_test,
        args.patches, args.band_patches
    )
    y_train, y_test, y_true = create_labels(number_train, number_test, number_true, num_classes)

    # Create data loaders (true/full-image loader removed — use chunked prediction instead)
    train_dataset = Data.TensorDataset(
        torch.from_numpy(x_train_band.transpose(0, 2, 1)).float(),
        torch.from_numpy(y_train).long()
    )
    test_dataset = Data.TensorDataset(
        torch.from_numpy(x_test_band.transpose(0, 2, 1)).float(),
        torch.from_numpy(y_test).long()
    )

    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    model = ViT(
        image_size = args.patches,
        near_band = args.band_patches,
        num_patches = band,
        num_classes = num_classes,
        dim = 64,
        depth = 5,
        heads = 4,
        mlp_dim = 8,
        dropout = 0.1,
        emb_dropout = 0.1,
        mode = args.mode
    )
    model = model.cuda()
    
    class_counts = torch.FloatTensor(number_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(number_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 10), gamma=args.gamma)

    if args.flag_train:
        print("Starting training...")
        tic = time.time()
        best_OA = 0.0
        best_epoch = 0
        epoch_history = []
        os.makedirs(config.save_model_path, exist_ok=True)
        save_path = os.path.join(config.save_model_path, 'vit_best_model.pt')

        for epoch in range(args.epochs):
            train_acc, train_loss, tar_t, pre_t = train_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()
            print(f"Epoch: {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            if epoch % args.test_freq == 0 or epoch == args.epochs - 1:
                tar_v, pre_v = valid_epoch(model, test_loader, criterion)
                OA, AA_mean, Kappa, AA, accuracy = calculate_metrics(tar_v, pre_v)
                print(f"  Val OA: {OA:.4f} | AA: {AA_mean:.4f} | Kappa: {Kappa:.4f}")
                epoch_history.append((epoch + 1, float(OA)))
                if OA > best_OA:
                    best_OA = OA
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), save_path)
                    print(f"  → Best model saved (OA: {best_OA:.4f})")

        toc = time.time()
        vit_train_time = toc - tic
        print(f"Training time: {vit_train_time:.2f} seconds")
        print(f"✅ Best model (OA={best_OA:.4f}) saved to: {save_path}")
        eval_dir = os.path.join(config.save_model_path, 'VITransformer_model_evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, 'vit_training_time.txt'), 'w') as f:
            f.write(f"ViT training time: {vit_train_time:.2f} s\n")
        with open(os.path.join(eval_dir, 'vit_epoch_history.txt'), 'w') as f:
            f.write("epoch,OA\n")
            for ep, oa in epoch_history:
                f.write(f"{ep},{oa:.4f}\n")
            f.write(f"best_epoch,{best_epoch}\n")
            f.write(f"best_OA,{best_OA:.4f}\n")
    
    # Training or testing
    if args.flag_test:
        model_path = os.path.join(config.save_model_path, 'vit_best_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        
        tar_v, pre_v = valid_epoch(model, test_loader, criterion)

        # Evaluation
        model_dir = config.save_model_path
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"✅ Created directory: {model_dir}")   
        evaluate_model(tar_v+1, pre_v+1, classname_dict=all_classes, save_dir=model_dir)

    if args.flag_predict:
        print("Running chunked full-image prediction...")
        pre_u = predict_full_image_chunked(
            model, mirror_image, total_pos_true,
            band, args.patches, args.band_patches
        )
        prediction_matrix = np.zeros((height, width), dtype=np.float32)
        for i in range(total_pos_true.shape[0]):
            prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1

        save_prediction_map(prediction_matrix, height, width, config)
        class_report = classification_report(tar_v, pre_v, digits=4)
        jaccard = jaccard_score(tar_v, pre_v, average=None)
        matrix = confusion_matrix(tar_v, pre_v)
        


   
    
    print("Final result:")
    # print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f} | Accuracy: {:.4f}".format(OA, AA_mean, Kappa, accuracy))
    print("**************************************************")
    print("Parameter:")

    def print_args(args):
        for k, v in zip(args.keys(), args.values()):
            print("{0}: {1}".format(k,v))

    print_args(vars(args))

    print("**************************************************")
    print("End of the program")
    print("**************************************************")
