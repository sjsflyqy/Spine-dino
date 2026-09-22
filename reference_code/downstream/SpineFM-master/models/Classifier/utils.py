
import torch
import torchvision
import numpy as np
import os
from skimage.measure import regionprops
import settings
from models.sam import sam_model_registry
from models.PointPredictor.point_predictor import PointPredictor


def get_model(name):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if name == 'Mask-RCNN':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn()

        # Update the model for 3 vertebrae classes + background class
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256 # default value
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

        # Load the trained weights to the model
        model.load_state_dict(torch.load(os.path.join('weights','mask_rcnn_2_categories.pth')))

        # Move the model to GPU if available        
        model.to(device)

        return model
    
    elif name == 'Medical-SAM-Adaptor':
        
        encoder = 'default'
        sam_checkpoint = os.path.join('weights','sam_vit_b_01ec64.pth')
        msa_checkpoint = os.path.join('weights','best_dice_checkpoint_centroid.pth')

        net = sam_model_registry[encoder](sam_checkpoint).to(device)

        '''load pretrained model'''
        checkpoint = torch.load(msa_checkpoint, device)
        new_state_dict = checkpoint['state_dict']

        net.load_state_dict(new_state_dict)
        return net
    
    elif name == 'MSA_estimate':
        
        encoder = 'default'
        sam_checkpoint = os.path.join('weights','sam_vit_b_01ec64.pth')
        msa_checkpoint = os.path.join('weights','best_dice_checkpoint.pth')

        net = sam_model_registry[encoder](sam_checkpoint).to(device)

        '''load pretrained model'''
        checkpoint = torch.load(msa_checkpoint, device)
        new_state_dict = checkpoint['state_dict']

        net.load_state_dict(new_state_dict)
        return net
    
    elif name == 'Point_Predictor':

        model = PointPredictor().to(device=device)
        model.load_state_dict(torch.load(os.path.join('weights','point_predictor.pth')))

        return model

    else:
        print('Invalid model name: ',name)
        exit()

def find_centroids(masks: torch.Tensor,input_scores: list,score_threshold=settings.MASK_RCNN_SCORE_THRESHOLD):
    # Input   masks: torch.Tensor with shape (N,1,H,W)
    #         input_scores: list of length N
    #
    # Output  centroids: list of centroids coordinates with length N
    #         scores: dictionary of centroid: score for each centroid

    centroids = []
    scores = {}

    for n,mask in enumerate(masks):
        
        # Extract centroid coordinates and their corresponding scores from output masks
        if input_scores[n] > score_threshold:
            thresholded_mask = (np.array(mask.squeeze(0)) > settings.MASK_RCNN_MASK_THRESHOLD).astype(np.uint8)
            c_y,c_x = regionprops(thresholded_mask)[0]['centroid']
            centroids.append((int(c_x),int(c_y)))
            scores[(int(c_x),int(c_y))] = input_scores[n]

    return centroids,scores

def find_centroids_weighted(masks: torch.Tensor,input_scores: list,score_threshold=settings.MASK_RCNN_SCORE_THRESHOLD):
    centroids = []
    scores = {}

    for n,mask in enumerate(masks):
        
        # reject any masks below settings.MASK_RCNN_SCORE_THRESHOLD
        if input_scores[n] > score_threshold:
            c_x,c_y = compute_weighted_centroid(np.array(mask.squeeze(0)))
            centroids.append((c_x,c_y))
            scores[c_x,c_y] = input_scores[n]
    
    return centroids,scores

def compute_weighted_centroid(mask):
    """
    Compute the centroid of a shape given a bitmap mask with probability weights.
    
    Parameters:
        mask (numpy.ndarray): 2D array where each pixel value is between 0 and 1.
        
    Returns:
        (float, float): The (x, y) coordinates of the centroid.
    """
    # Get the shape of the mask
    rows, cols = mask.shape
    
    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Flatten the coordinate grids and mask
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    weights_flat = mask.flatten()
    
    # Compute weighted sums
    total_weight = np.sum(weights_flat)
    weighted_x_sum = np.sum(x_flat * weights_flat)
    weighted_y_sum = np.sum(y_flat * weights_flat)
    
    # Compute the centroid coordinates
    centroid_x = int(weighted_x_sum / total_weight)
    centroid_y = int(weighted_y_sum / total_weight)
    
    return centroid_x, centroid_y


def generate_patch(img,center,patch_size):
    # extracts patch from base_img and base_gt around the center

    base_w,base_h = img.size
    c_x,c_y = center

    # apply padding if needed
    pad = [0,0,0,0]
    flag = False
    
    # left padding
    if c_x < patch_size/2:
        flag = True
        pad[0] = patch_size/2 - c_x
    # right padding
    elif base_w-c_x < patch_size/2:
        flag = True
        pad[2] = patch_size/2 - base_w + c_x
    # top padding
    if c_y < patch_size/2:
        flag = True
        pad[1] = patch_size/2 - c_y
    # bottom padding
    elif base_h-c_y < patch_size/2:
        flag = True
        pad[3] = patch_size/2 - base_h + c_y
    if flag:
        pad = (int(pad[0]),int(pad[1]),int(pad[2]),int(pad[3]))
        torchvision.transforms.functional.pad(img,pad)
    
    # crop image and ground truth to region of interest
    top = int(c_y - patch_size/2)
    left = int(c_x - patch_size/2)
    
    cropped_img = torchvision.transforms.functional.crop(img,top=top,left=left,height=patch_size,width=patch_size)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((1024,1024)),
        torchvision.transforms.ToTensor(),
    ])

    return transform(cropped_img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def reverse_patch(mask,center,original_size,patch_size):
    # transform mask frame of reference from its patch to the original image

    mask = torch.nn.functional.interpolate(mask,size=(patch_size,patch_size)).squeeze(0)
    c_x,c_y = (round(center[0]),round(center[1]))
    patch_size = mask.shape[-1]
    fill = int(np.min(np.array(mask.cpu())))

    pad = [int(c_x-patch_size/2),int(c_y-patch_size/2),int(original_size[1]-c_x-patch_size/2),int(original_size[0]-c_y-patch_size/2)]
    mask = torchvision.transforms.functional.pad(mask,pad,fill=fill)
    return mask

def MSA_predict(MSA,img,point,patch_size):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    patch = generate_patch(img,point,patch_size=patch_size).unsqueeze(0).to(dtype=torch.float32,device=device)
    pt = (torch.Tensor([[[512,512]]]).to(dtype=torch.float,device=device),torch.Tensor([[1]]).to(dtype=torch.int,device=device))

    with torch.no_grad():
        imge = MSA.image_encoder(patch)
        se, de = MSA.prompt_encoder(
            points=pt,
            boxes=None,
            masks=None,
        )
        pred, score = MSA.mask_decoder(
            image_embeddings=imge,
            image_pe=MSA.prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de, 
            multimask_output=False,
        )
    
    return pred,score.cpu().item()

def iou_from_logits(logits1, logits2, threshold=0.5):
    # Convert logits to binary masks
    pred_mask = (logits1 > threshold).float()
    gt_mask = (logits2 > threshold).float()

    # Calculate intersection and union
    intersection = torch.sum(pred_mask * gt_mask)
    union = torch.sum(pred_mask + gt_mask) - intersection

    # Handle edge case where union is 0
    if union == 0:
        return torch.tensor(1.0)  # Both masks are empty, consider IoU as 1.0
    
    # Calculate IoU
    iou = intersection / union
    return iou

def random_click(mask):
    # max agreement position
    indices = np.argwhere(mask == 1) 
    return tuple(indices[np.random.randint(len(indices))])