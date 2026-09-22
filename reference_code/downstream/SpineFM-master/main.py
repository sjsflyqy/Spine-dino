import numpy as np
import os
import sys
from PIL import Image
import torch
import pickle
from dataset import *
from utils import *

def main():

    with torch.no_grad():
        
        if len(sys.argv) != 5:
            sys.exit("Usage: python main.py \"output_directory\" \"weights_path\" \"dataset\" \"data_path\"")
        
        # name of output directory
        output_directory = sys.argv[1]

        # path to model weights
        weights_path = sys.argv[2]

        # dataset (NHANESII or CSXA)
        dataset = sys.argv[3]

        # data path
        data_path = sys.argv[4]

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # these should match the patch sizes and class numbers used during training
        if dataset == 'NHANESII':
            MSA_patch_size = 600
            Classifier_patch_size = 512
            num_classes = 4
        elif dataset == 'CSXA':
            MSA_patch_size = 300
            Classifier_patch_size = 256
            num_classes = 2
        else:
            sys.exit("Invalid dataset. Please choose either NHANESII or CSXA.")

        # ------------------------------------------- PRELIMINARY MASK GENERATION --------------------------------------------

        # This section runs the original images through a trained Mask-RCNN, generating candidate vertebrae masks as an 
        # output, along with their corresponding scores (i.e. the model's confidence) and the ground truth.

        print("Generating preliminary masks. \n")


        # Load the Mask-RCNN model
        Mask_RCNN = get_model('Mask-RCNN',device,weights_path)

        # Get the data_loader
        data_loader = get_data_loader(dataset=dataset,data_path=data_path,mode='Testing',b=1)

        masks_dict = {}
        scores_dict = {}

        Mask_RCNN.eval()
        for images, targets in data_loader:

            ids = [target['image_id'] for target in targets]

            # Run the prediction
            images = [image.to(device) for image in images]
            outputs = Mask_RCNN(images)

            # Organise outputs for further processing
            for i in range(len(outputs)):
                masks_dict[ids[i]] = outputs[i]['masks'].cpu()
                scores_dict[ids[i]] = outputs[i]['scores'].cpu()
            
        del Mask_RCNN

        # ------------------------------------------- INTIAL MSA PROMPT GENERATION -------------------------------------------

        # This section processes output masks into the initial point prompts for Medical-SAM-Adaptor. It achieves this by:

        # -> filtering masks with low score
        # -> thresholding the output masks to generate a binary bitmap
        # -> locating the centroid of each vertebrae mask
        # -> sorting the centroids into their spacial order along the spine
        # -> chosing the three sequential centroids with the highest average score

        print("Calculating vertebrae prompts. \n")

        top_three_dict = {}
        pred_centroids = {}

        for id in masks_dict.keys():

            masks = masks_dict[id]
            scores = scores_dict[id]

            # generate centroids coordinates for reasonable scoring masks
            centroids,scores = find_centroids(masks,scores,0.6)

            # fit a line along these centroids
            x,y = zip(*centroids)
            A = np.array([x,np.ones_like(x)]).T
            m,c = np.linalg.inv((A.T@A))@(A.T@np.array(y))

            # sort the list of centroids by their y projection onto the line of best fit
            y_proj = m*(x+m*(y-c))/(1+m**2)+c
            centroids = [centroids[i] for i in np.argsort(y_proj)]

            # find the 3 sequential centroids with the top average score
            top_avg = 0
            for i in range(len(centroids)-2):
                avg_score = sum((scores[centroids[i]],scores[centroids[i+1]],scores[centroids[i+2]]))/3
                if avg_score > top_avg:
                    top_avg = avg_score
                    top_three = (centroids[i],centroids[i+1],centroids[i+2])
            
            top_three_dict[id] = top_three
            pred_centroids[id] = centroids

        # --------------------------------- INITIALISE MSA MODEL AND SET UP PREDICTION LOOP ----------------------------------

        # This section initialises the Medical-SAM-Adaptor model, as well as running the initial prompts through the model to 
        # refine the first known centroids.

        print("Setting up prediction loop. \n")

        # Load the Medical-SAM-Adaptor model
        MSA = get_model('Medical-SAM-Adaptor',device,weights_path)
        MSA.eval()

        # Load a ResNet based model used for classifying new vertebra predictions (between background, regular vertebra, and 
        # landmark vertebra)
        Classifier = get_model('ResNet_Classifier',device,weights_path,num_classes=num_classes)
        Classifier.eval()

        refined_points = {}
        masks = {}
        landmark_centroids = {id:(0,0) for id in top_three_dict.keys()}

        # Predict the first three vertebrae masks from the output prompts of the previous step
        for id,points in top_three_dict.items():

            img = Image.open(os.path.join(data_path,'imgs',id+'.jpg')).convert('RGB')
            output_masks = []
            
            for point in points:
            
                pred = MSA_predict(MSA,img,point,patch_size=MSA_patch_size)
                output_masks.append(pred)
                p = compute_weighted_centroid(np.array(torch.sigmoid(pred).squeeze(0)))

                ## labelling only works for NHANESII for now
                if num_classes > 2:
                    label = classify(Classifier,img,p,patch_size=Classifier_patch_size)

                    # record the centroid of the S1 vertebra
                    if label == 3:
                        landmark_centroids[id] = p
                    
                    # record the centroid of the C2 vertebra
                    elif label == 2:
                        landmark_centroids[id] = p

            
            # Recalculate vertebrae centroids from MSA generated masks
            refined_points[id] = [compute_weighted_centroid(np.array(torch.sigmoid(mask).squeeze(0))) for mask in output_masks] 
            masks[id] = output_masks

        # ----------------------------------------- MAIN VERTEBRAE PREDICTION LOOP -------------------------------------------

        # This main loop segments the remaining vertebrae within the X-ray. It does this by sequentially trying to segment the
        # next vertebra in the chain, and then testing to see if the mask generated indicates that a new vertebra has actually
        # been found. For more indication of the specifics read the code.

        print("Generating masks... \n")

        # Load a simple fully connected NN which predicts the location of the centroid of the next vertebra
        PointPredictor = get_model('Point_Predictor',device,weights_path)

        os.makedirs(os.path.join(data_path,output_directory),exist_ok=True)
        os.makedirs(os.path.join(data_path,output_directory,'extras'),exist_ok=True)
        os.makedirs(os.path.join(data_path,output_directory,'masks'),exist_ok=True)

        for n,(id,p) in enumerate(refined_points.items()):
            
            if n != 0 and n % 10 == 0:
                print(n,'/',len(refined_points),'completed.')
            
            # Load the X-ray image corresoponding to id
            img = Image.open(os.path.join(data_path,'imgs',id+'.jpg')).convert('RGB')
            w,h = img.size

            new_points = []
            new_points_refined = []
            extras = {}

            # By iterating over both the forward and reverse of points we can predict new vertebra both up and down the spine
            for points in [p[::-1],p]:
                
                # Ensure that the ordering of the masks list matches the order of the points list 
                masks[id] = masks[id][::-1]

                while True:
                    
                    # Normalise and format the points so they can be input into the point predictor model
                    normalised_points = [(x/w,y/h) for (x,y) in points]
                    normalised_points = torch.tensor(normalised_points, dtype=torch.float32, device=device)

                    # Predict the location of the next vertebra from the centroids of the previous three
                    new_point = PointPredictor(normalised_points.view(-1,6))

                    # Rescale the prediciton back to the base image size
                    new_point = np.array(new_point.cpu().squeeze(0))*img.size
                    new_points.append(tuple(new_point))

                    # Predict the mask of the next vertebra by prompting the predicted centroid into MSA
                    new_mask = MSA_predict(MSA,img,new_point,patch_size=MSA_patch_size)

                    # Check that the mask isn't overlapping with the previous mask (especially problematic for C2)
                    iou = iou_from_logits(new_mask,masks[id][-1])
                    if iou > 0.1:
                        # If it is then do not save the new mask and stop the loop
                        break
                    
                    # Calculate a more refined position of the new vertebra centroid
                    new_point_refined = compute_weighted_centroid(np.array(torch.sigmoid(new_mask).squeeze(0)))

                    # Use the ResNet classifier to predict the label of the new region
                    label = classify(Classifier,img,new_point_refined,patch_size=Classifier_patch_size)

                    # Background: Stop loop and discard mask
                    if label == 0:
                        break
                    
                    # S1: Stop loop and save mask
                    elif label == 3:
                        new_points_refined.append(new_point_refined)
                        landmark_centroids[id] = new_point_refined
                        masks[id].append(new_mask)
                        break
                    
                    # C2: Stop loop and save mask
                    elif label == 2:
                        new_points_refined.append(new_point_refined)
                        landmark_centroids[id] = new_point_refined
                        masks[id].append(new_mask)
                        break
                    
                    # Regular vertebra: Continue loop and save mask
                    else:
                        new_points_refined.append(new_point_refined)
                        masks[id].append(new_mask)

                        # Replace the furthest point with the new vertebra's centroid
                        points = points[1:]+[new_point_refined]
            
            # Post process by thresholding the masks then gaussian smoothing (only helpful for NHANES II)
            if dataset == 'NHANESII':
                masks[id] = np.array([gaussian_smooth(binary_mask_from_logits(mask,0.9)) for mask in masks[id]])

            # Save the masks
            np.savez_compressed(os.path.join(data_path,output_directory,'masks',id+'.npz'),masks=masks[id])

            # Save additional information
            with open(os.path.join(data_path,output_directory,'extras',id+'.pkl'),'wb') as f:

                # Save extra data which could be useful
                extras['landmark_centroid'] = landmark_centroids[id]
                extras['rough_points'] = new_points
                extras['refined_points'] = new_points_refined
                extras['starting_points'] = top_three_dict[id]
                extras['refined_starting_points'] = refined_points[id]

                pickle.dump(extras,f)

    print("\nDone!\n")

if __name__ == '__main__':
    main()