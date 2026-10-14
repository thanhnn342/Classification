import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # Set the GPU 1 to use
import torch 
import numpy as np
from helper import get_sample, get_loader, conv_image
from Model import ResNet, AlexNet, GoogLeNet, MobileNet
import cv2 as cv 
from tqdm import tqdm 

def main(model, data_path, num_test, ckpt_path, device, res_path, save_image = True, epoch = 0):

    os.makedirs(os.path.join(res_path, f'{model.name}'), exist_ok=True)

    data_path_train = os.path.join(data_path, 'train')
    data_path_test = os.path.join(data_path, 'test')

    train_data, test_data = get_sample(data_path_train = data_path_train
                                         , data_path_validation = data_path_test)

    trainloader, testloader = get_loader(train_data = train_data,
                                        validation_data = test_data,
                                        BATCH_SIZE = 1)
    
    label_name = sorted(os.listdir(data_path_test))

    if epoch == 0:
        check_point = torch.load(os.path.join(ckpt_path,f"Best_ckpt_{model.name}.pt"), map_location=device)
    else:
        check_point = torch.load(os.path.join(ckpt_path,f"{model.name}_{epoch}_epoch.pt"), map_location=device)

    model.load_state_dict(check_point['state_dict'])

    print('-- Number of Sample Test:', len(test_data))
    print('-- Model use:', model.name)

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.inference_mode():
        pbar = tqdm(testloader)
        
        for i, (image, label) in enumerate(pbar):
            
            if i >= num_test:
                break

            image = image.to(device)
            label = label.to(device)

            output = model(image)
            _, predicted = torch.max(output.data, 1)

            true_labels.append(label.item())
            predicted_labels.append(predicted.item())

            if save_image:
                image = (conv_image(image) * 255).astype(np.uint8)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                save_path = os.path.join(res_path, 
                                            f'{model.name}', 
                                            f'{i:04d}_True_label: {label_name[label.item()]}, Predicted_label: {label_name[predicted.item()]}.png')
                
                cv.imwrite(save_path, image)

            pbar.set_description(
                f"Processing: [{i+1}/{num_test}]"
            )

        true_labels = np.asarray(true_labels)
        predicted_labels = np.asarray(predicted_labels)
        accuracy = (predicted_labels == true_labels).mean()

    print(f"Complete Testing {model.name} !!!")
    print(f"Overall Accuracy: {accuracy}")

if __name__ == "__main__":

    import argparse 

    MODELS = {"resnet": ResNet.ResNet, "alexnet": AlexNet.AlexNet,
          "mobilenet": MobileNet.MobileNet, "googlenet": GoogLeNet.GoogLeNet}

    parser = argparse.ArgumentParser()

    parser.add_argument("--model",choices=MODELS.keys(), default="resnet", help= "Determine which type of model to use (resnet, alexnet, googlenet, mobilenet)")
    parser.add_argument("--data_path", required = True, help= "Determine the path to the dataset")
    parser.add_argument("--n_test", type = int, default=100, help= "Determine the number of Epoch (Default 100)")
    parser.add_argument("--ckpt_path", type = str, required=True, help= "Checkpoint path")
    parser.add_argument("--save_image", action="store_true", help= "Save visualize images")
    parser.add_argument("--res_path", type = str, default='result', help= "Result path")
    parser.add_argument("--num_epoch_test", type = int, default=0, help= "Choose specific epoch to evaluate")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('-- Using device:',device)

    data_path = args.data_path
    num_class = len(os.listdir(os.path.join(data_path, 'train')))
    print('-- Number of Classes:', num_class)

    print("-- Num_test: ", args.n_test)

    model = MODELS[args.model](num_classes=num_class)
    model.to(device)

    main(model=model,
         data_path=data_path,
         num_test=args.n_test,
         ckpt_path=args.ckpt_path,
         save_image=args.save_image,
         res_path=args.res_path,
         device=device,
         epoch=args.num_epoch_test)