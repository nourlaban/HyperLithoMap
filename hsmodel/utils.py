import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
import rasterio
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from timm.loss import LabelSmoothingCrossEntropy
import numpy as np
import os
import sys
# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projutils.config import config
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns



def setup_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def load_HSI(args):
    if args.dataset == 'Indian':
        data = loadmat('./data/IndianPine.mat')
    elif args.dataset == 'Pavia':
        data = loadmat('./data/Pavia.mat')
    elif args.dataset == 'Houston':
        data = loadmat('./data/Houston.mat')
    elif args.dataset == 'WHU_Hi_LongKou':
        data = loadmat('./data/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        TR = loadmat('./data/WHU-Hi-LongKou/TRTE/Train25.mat')['LKtrain25']
        TE = loadmat('./data/WHU-Hi-LongKou/TRTE/Test25.mat')['LKtest25']

    elif args.dataset == 'aboresheed':
        with rasterio.open(config.tifpath ) as src:
            input_data = src.read()
        input = np.transpose(input_data, (1, 2, 0))   

        with rasterio.open(config.train_mask_path) as src: 
            TR = src.read(1)
        with rasterio.open(config.test_mask_path) as src: 
            TE = src.read(1)      
       

    else:
        raise ValueError("Unkknow dataset")
    # if not args.dataset == 'WHU_Hi_LongKou':
    #     TR = data['TR']
    #     TE = data['TE']
    #     input = data['input']
    # else:
    #     input = data
        
    label = TR + TE # type: ignore
    num_classes = np.max(TR)
    _NODATA = -9999.0
    input_normalize = np.zeros(input.shape) # type: ignore
    for i in range(input.shape[2]): # type: ignore
        band = input[:,:,i]
        valid = band[band != _NODATA]
        if len(valid) > 0 and valid.max() > valid.min():
            input_normalize[:,:,i] = np.where(
                band != _NODATA,
                (band - valid.min()) / (valid.max() - valid.min()),
                0.0
            )
    color_matrix = [
                    [0,0,0],
                    [83,171,72],
                    [137,186,67],
                    [66,132,91],
                    [60,131,69],
                    [144,82,54],
                    [105,188,200],
                    [255,255,255],
                    [199,176,201],
                    [218,51,44],
                    [119,35,36],
                    [55,101,166],
                    [224,219,84],
                    [217,142,52],
                    [84,48,126],
                    [227,119,91],
                    [157,87,150]
                    ]
    color_matrix_pred = [
                    [83,171,72],
                    [137,186,67],
                    [66,132,91],
                    [60,131,69],
                    [144,82,54],
                    [105,188,200],
                    [255,255,255],
                    [199,176,201],
                    [218,51,44],
                    [119,35,36],
                    [55,101,166],
                    [224,219,84],
                    [217,142,52],
                    [84,48,126],
                    [227,119,91],
                    [157,87,150]
                    ]
    color_matrix = [[c/256. for c in color] for color in color_matrix] 
    color_matrix_pred = [[c/256. for c in color] for color in color_matrix_pred]
    return input_normalize, label, num_classes, TR, TE, color_matrix, color_matrix_pred
    
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] 
    total_pos_train = total_pos_train.astype(int)

    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class
    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] 
    total_pos_test = total_pos_test.astype(int)

    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class
    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)
    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true

def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]
    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi

def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image


def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    # x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    # for k in range(true_point.shape[0]):
    #     x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    # print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    return x_train, x_test#, x_true#

def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()


def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   
        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()       
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()  
        batch_pred = model(batch_data)
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return tar, pre

def predict_whole_image(model, height, width, band, mirror_image, args,
                       input_raster_path=None,
                       output_path=None):
    """
    Predict labels for an entire image by processing it in patches and save as georeferenced TIFF.
    
    Args:
        model: The trained model for prediction
        height: Height of the image
        width: Width of the image
        band: Number of bands/channels in the image
        mirror_image: The input image data (numpy array)
        args: Additional arguments (should contain patch_size, stride, batch_size, etc.)
        input_raster_path: Path to input raster for copying georeferencing (default: vnir_swir_stackedd.tif)
        output_path: Path where the output GeoTIFF will be saved
    
    Returns:
        pre_map: Predicted label map for the entire image
    """
    import torch
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from torch.nn import functional as F
    import os

    # Resolve default paths at call time so they follow the active dataset.
    input_raster_path = input_raster_path or config.tifpath
    output_path = output_path or config.output_MAMBA_tif_path

    # Set model to evaluation mode
    model.eval()
    
    # Get patch parameters from args
    patch_size = args.patches  # Default patch size
    stride = patch_size
    batch_size = getattr(args, 'batch_size', 4)
    
    # Initialize prediction map
    pre_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)  # For averaging overlapping predictions
    
    # Calculate number of patches
    patches = []
    positions = []

    _NODATA = -9999.0
    input_normalize = np.zeros(mirror_image.shape) # type: ignore
    for i in range(mirror_image.shape[2]): # type: ignore
        b = mirror_image[:,:,i]
        valid = b[b != _NODATA]
        if len(valid) > 0 and valid.max() > valid.min():
            input_normalize[:,:,i] = np.where(
                b != _NODATA,
                (b - valid.min()) / (valid.max() - valid.min()),
                0.0
            )
    
    # Extract patches with sliding window
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = input_normalize[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)
            positions.append((y, x))
    
    # Handle edge cases - add patches for right and bottom edges
    # Right edge
    if width % stride != 0:
        for y in range(0, height - patch_size + 1, stride):
            x = width - patch_size
            patch = input_normalize[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)
            positions.append((y, x))
    
    # Bottom edge
    if height % stride != 0:
        for x in range(0, width - patch_size + 1, stride):
            y = height - patch_size
            patch = input_normalize[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)
            positions.append((y, x))
    
    # Bottom-right corner
    if height % stride != 0 and width % stride != 0:
        y = height - patch_size
        x = width - patch_size
        patch = input_normalize[y:y+patch_size, x:x+patch_size, :]
        patches.append(patch)
        positions.append((y, x))
    
    # Convert patches to tensor
    patches = np.array(patches)
    
    # Process patches in batches
    num_patches = len(patches)
    
    with torch.no_grad():
        for i in range(0, num_patches, batch_size):
            batch_patches = patches[i:min(i+batch_size, num_patches)]
            batch_positions = positions[i:min(i+batch_size, num_patches)]
            
            # Convert to torch tensor and move to GPU
            batch_data = torch.from_numpy(batch_patches).float()
            
            # Adjust dimensions if needed (assuming model expects [B, C, H, W])
            if len(batch_data.shape) == 4 and batch_data.shape[-1] == band:
                batch_data = batch_data.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            
            batch_data = batch_data.cuda()
            
            # Get predictions
            batch_pred = model(batch_data)
            
            # Handle different output formats
            if isinstance(batch_pred, tuple):
                batch_pred = batch_pred[0]
            
            # Get predicted classes (assuming classification task)
            if len(batch_pred.shape) > 2:  # If output is segmentation map
                # For semantic segmentation
                pred_classes = torch.argmax(batch_pred, dim=1).cpu().numpy()
            else:
                # For patch-level classification
                pred_classes = torch.argmax(batch_pred, dim=1).cpu().numpy()
                # Expand to patch size
                pred_classes = np.repeat(pred_classes[:, np.newaxis, np.newaxis], 
                                        patch_size, axis=1)
                pred_classes = np.repeat(pred_classes, patch_size, axis=2)
            
            # Place predictions in the output map
            for j, (y, x) in enumerate(batch_positions):
                if len(pred_classes.shape) == 3:
                    pre_map[y:y+patch_size, x:x+patch_size] += pred_classes[j]
                else:
                    pre_map[y:y+patch_size, x:x+patch_size] += pred_classes[j, :patch_size, :patch_size]
                count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping predictions
    pre_map = np.divide(pre_map, count_map, where=count_map!=0)
    
    # Round to get integer class labels
    pre_map = np.round(pre_map).astype(np.uint8)  # Using uint8 for class labels
    
    # Save as GeoTIFF with georeferencing from vnir_swir_stackedd.tif
    try:
        # Check if input raster path exists
            with rasterio.open(config.tifpath) as src:
                profile = src.profile
                profile.update(dtype=rasterio.int16, count=1, compress='lzw')

            with rasterio.open(config.output_MAMBA_tif_path, 'w', **profile) as dst:   
                    dst.write(pre_map+1, 1)      
                  
                    
                    # Optionally add color interpretation for visualization
                    try:
                        dst.write_colormap(
                            1, {
                                0: (0, 0, 0, 255),       # Class 0: Black
                                1: (255, 0, 0, 255),     # Class 1: Red
                                2: (0, 255, 0, 255),     # Class 2: Green
                                3: (0, 0, 255, 255),     # Class 3: Blue
                                4: (255, 255, 0, 255),   # Class 4: Yellow
                                5: (255, 0, 255, 255),   # Class 5: Magenta
                                6: (0, 255, 255, 255),   # Class 6: Cyan
                                7: (128, 128, 128, 255), # Class 7: Gray
                                8: (255, 128, 0, 255),   # Class 8: Orange
                                9: (128, 0, 255, 255),   # Class 9: Purple
                                # Add more colors as needed for your classes
                            }
                        )
                    except:
                        # Skip colormap if not supported
                        pass
                
                    print(f"Prediction map saved with georeferencing from {input_raster_path}")
                    print(f"Output saved to: {output_path}")
                    print(f"CRS: {profile['crs']}")
                    print(f"Transform: {profile['transform']}")
        
            
    except Exception as e:
        print(f"Warning: Could not copy georeferencing from {input_raster_path}: {e}")
        print("Creating basic georeferenced output...")
        
        # Create basic georeferencing if unable to copy from input
        transform = from_origin(0, height, 1, 1)  # (west, north, xsize, ysize)
        
        meta = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 1,
            'width': width,
            'height': height,
            'transform': transform,
            'crs': 'EPSG:4326',  # Default to WGS84
            'compress': 'lzw',
            'nodata': 255
        }
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(pre_map, 1)
        
        print(f"Prediction map saved with default georeferencing to: {output_path}")
    
    return pre_map


def predict_whole_image_with_probability(model, height, width, band, mirror_image, args, 
                                        input_raster_path=None, output_path='prediction2.tif',
                                        save_probability=False, prob_output_path='probability.tif'):
    """
    Extended version that can also save probability maps for each class.
    
    Additional Args:
        save_probability: Whether to save probability maps
        prob_output_path: Path for probability maps output
    """
    import torch
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from torch.nn import functional as F
    
    # Set model to evaluation mode
    model.eval()
    
    # Get patch parameters from args
    patch_size = args.patches
    stride = patch_size
    batch_size = getattr(args, 'batch_size', 4)
    
    # Determine number of classes (you might need to adjust this based on your model)
    num_classes = getattr(args, 'num_classes', None)
    
    # Initialize prediction maps
    pre_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)
    
    # Initialize probability map if needed
    if save_probability and num_classes:
        prob_map = np.zeros((num_classes, height, width), dtype=np.float32)
        prob_count_map = np.zeros((height, width), dtype=np.float32)
    
    # Calculate patches
    patches = []
    positions = []
    
    # Extract patches with sliding window
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = mirror_image[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)
            positions.append((y, x))
    
    # Handle edges
    if width % stride != 0:
        for y in range(0, height - patch_size + 1, stride):
            x = width - patch_size
            patch = mirror_image[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)
            positions.append((y, x))
    
    if height % stride != 0:
        for x in range(0, width - patch_size + 1, stride):
            y = height - patch_size
            patch = mirror_image[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)
            positions.append((y, x))
    
    if height % stride != 0 and width % stride != 0:
        y = height - patch_size
        x = width - patch_size
        patch = mirror_image[y:y+patch_size, x:x+patch_size, :]
        patches.append(patch)
        positions.append((y, x))
    
    patches = np.array(patches)
    num_patches = len(patches)
    
    with torch.no_grad():
        for i in range(0, num_patches, batch_size):
            batch_patches = patches[i:min(i+batch_size, num_patches)]
            batch_positions = positions[i:min(i+batch_size, num_patches)]
            
            batch_data = torch.from_numpy(batch_patches).float()
            
            if len(batch_data.shape) == 4 and batch_data.shape[-1] == band:
                batch_data = batch_data.permute(0, 3, 1, 2)
            
            batch_data = batch_data.cuda()
            
            batch_pred = model(batch_data)
            
            if isinstance(batch_pred, tuple):
                batch_pred = batch_pred[0]
            
            # Get probabilities if needed
            if save_probability and num_classes:
                batch_prob = F.softmax(batch_pred, dim=1).cpu().numpy()
            
            # Get predicted classes
            pred_classes = torch.argmax(batch_pred, dim=1).cpu().numpy()
            
            # Place predictions
            for j, (y, x) in enumerate(batch_positions):
                pre_map[y:y+patch_size, x:x+patch_size] += pred_classes[j]
                count_map[y:y+patch_size, x:x+patch_size] += 1
                
                if save_probability and num_classes:
                    for c in range(num_classes):
                        prob_map[c, y:y+patch_size, x:x+patch_size] += batch_prob[j, c]
    
    # Average overlapping predictions
    pre_map = np.divide(pre_map, count_map, where=count_map!=0)
    pre_map = np.round(pre_map).astype(np.uint8)
    
    if save_probability and num_classes:
        for c in range(num_classes):
            prob_map[c] = np.divide(prob_map[c], count_map, where=count_map!=0)
    
    # Get georeferencing information
    if input_raster_path and input_raster_path != '':
        with rasterio.open(input_raster_path) as src:
            meta = src.meta.copy()
            transform = src.transform
            crs = src.crs
    else:
        transform = from_origin(0, height, 1, 1)
        crs = 'EPSG:4326'
        meta = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'transform': transform,
            'crs': crs
        }
    
    # Save prediction map
    meta.update({
        'dtype': 'uint8',
        'count': 1,
        'compress': 'lzw',
        'nodata': 255
    })
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(pre_map, 1)
    
    # Save probability maps if requested
    if save_probability and num_classes:
        meta.update({
            'dtype': 'float32',
            'count': num_classes,
            'nodata': -9999
        })
        
        with rasterio.open(prob_output_path, 'w', **meta) as dst:
            for c in range(num_classes):
                dst.write(prob_map[c], c + 1)
                dst.set_band_description(c + 1, f'Class_{c}_probability')
    
    print(f"Prediction map saved to: {output_path}")
    if save_probability:
        print(f"Probability map saved to: {prob_output_path}")
    
    return pre_map

def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()  
        batch_pred = model(batch_data)
        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


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
            eval_path = os.path.join(save_dir, 'mamba_model_evaluation')
            os.makedirs(eval_path, exist_ok=True)
            plt.savefig(os.path.join(eval_path, 'mamba_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved")

            # Write results to text file
            results_path = os.path.join(eval_path, 'mamba_evaluation_results.txt')
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

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
