import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 1 to use

import torch 
from tqdm import tqdm 
from helper import get_sample, get_loader, save_model, save_plot, init_training
from torch.nn import CrossEntropyLoss
from Model import ResNet, AlexNet, GoogLeNet, MobileNet

def main(model, data_path, BATCH_SIZE, learning_rate, NUM_EPOCHS, save_every, save_path, device):

    data_path_train = os.path.join(data_path, 'train')
    data_path_validation = os.path.join(data_path, 'validation')

    train_data, validation_data = get_sample(data_path_train = data_path_train
                                         , data_path_validation = data_path_validation)

    trainloader, validationloader = get_loader(train_data = train_data,
                                        validation_data = validation_data,
                                        BATCH_SIZE = BATCH_SIZE)
    
    print('-- Number of Sample Train:', len(train_data))
    print('-- Number of Sample Validation:', len(validation_data))
    print('-- Training model:', model.name)

    save_path, logFile = init_training(model = model, save_path=save_path)

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses_train = []
    losses_validation = []

    acc_train = []
    acc_validation = []

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(trainloader)

        model.train()
        train_avg = 0

        total_label = 0 
        correct_label = 0

        for i, (image, label) in enumerate(pbar):

            image = image.to(device)
            label = label.to(device)

            predicted = model(image)
            _, output = torch.max(predicted.data, 1)

            train_val = criterion(predicted, label)

            optimizer.zero_grad()
            train_val.backward()
            optimizer.step()

            pbar.set_description(
                f"Epoch: [{epoch+1}/{NUM_EPOCHS}], Loss_Val: {train_val.item():.4f}"
            )

            train_avg += train_val.item()

            total_label += label.size(0)
            correct_label += (output == label).sum().item()
                
        train_avg /= len(trainloader)
        losses_train.append(train_avg)
        acc_train.append(correct_label/total_label)

        model.eval()

        with torch.inference_mode():
            validation_avg = 0

            total_label = 0 
            correct_label = 0

            for i, (image, label) in enumerate(validationloader):
                image = image.to(device)
                label = label.to(device)

                predicted = model(image)
                _, output = torch.max(predicted.data, 1)

                validation_val = criterion(predicted, label)

                validation_avg += validation_val.item()

                total_label += label.size(0)
                correct_label += (output == label).sum().item()
                
            validation_avg /= len(validationloader)
            losses_validation.append(validation_avg)
            acc_validation.append(correct_label/total_label)

        save_model(epoch=epoch+1,
                model=model,
                optimizer=optimizer,
                train_losses=losses_train,
                validation_losses=losses_validation,
                save_path=save_path, 
                save_every=save_every,
                save_best = True)
        
        save_plot(epoch=epoch+1,
                losses_train=losses_train,
                losses_validation=losses_validation,
                acc_train=acc_train,
                acc_validation=acc_validation,
                save_path=save_path)

    with open(logFile, "w") as file:    
        file.write("Loss value \n")
        file.write(f'Train: {losses_train} \n')
        file.write(f'Validation: {losses_validation} \n')

        file.write("Accuracy value \n")
        file.write(f'Train: {acc_train} \n')
        file.write(f'Validation: {acc_validation} \n')

    print(f"Complete Training {model.name} !!!")

if __name__ == "__main__":

    import argparse 
    MODELS = {"resnet": ResNet.ResNet, "alexnet": AlexNet.AlexNet,
          "mobilenet": MobileNet.MobileNet, "googlenet": GoogLeNet.GoogLeNet}

    parser = argparse.ArgumentParser()

    parser.add_argument("--model",choices=MODELS.keys(), default="resnet", help= "Determine which type of model to use (resnet, alexnet, googlenet, mobilenet)")
    parser.add_argument("--data_path", required = True, help= "Determine the path to the dataset")
    parser.add_argument("--batch_size", type = int, default=64, help= "Determine the number of batchsize (Default 64)")
    parser.add_argument("--lr", type = float,  default=1e-4, help= "Determine the learning rate (Default 1e-4)")
    parser.add_argument("--n_epochs", type = int, default=100, help= "Determine the number of Epoch (Default 100)")
    parser.add_argument("--save_every", type = int, default=2, help= "Determine to save the model every each epoch (Default 2)")
    parser.add_argument("--ckpt_path", type = str, default=None, help= "Checkpoint_path")
    parser.add_argument("--seed", type = int, default=None, help= "Torch random seed")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('-- Using device:',device)

    if args.seed is not None: torch.manual_seed(args.seed) 
    print('-- Using seed:', args.seed)

    data_path = args.data_path

    num_class = len(os.listdir(os.path.join(data_path, 'train')))
    print('-- Number of Classes:', num_class)

    print("-- Batch_size: ", args.batch_size)
    print("-- Learning_rate: ", args.lr)
    print("-- Num_Epochs: ", args.n_epochs)
    print("-- Save_every: ", args.save_every)

    model = MODELS[args.model](num_classes=num_class)
    model.to(device)

    main(model=model,
         data_path=data_path,
         BATCH_SIZE=args.batch_size, 
         learning_rate=args.lr,
         NUM_EPOCHS=args.n_epochs,
         save_every=args.save_every,
         save_path=args.ckpt_path,
         device=device)