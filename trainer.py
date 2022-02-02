import random

import torch
from torchattacks import FGSM, RFGSM, MIFGSM


def identity(woof, *args, **kwargs):
    return woof


def epoch_train(model, training_data_loader, data_augmentation, optimiser, criterion, training_device,
                adv_training, ensemble_list, l_ep, u_ep):
    for i, (x_input, y_target) in enumerate(training_data_loader):

        if adv_training:
            if ensemble_list:
                adv_model = random.choice(ensemble_list)
            else:
                adv_model = model
            # mi_steps = 5
            # epsilon = l_ep + (u_ep - l_ep) * random.random()
            attack = random.choice([
                MIFGSM(adv_model),
                identity,
                identity,
                FGSM(adv_model),
                identity,
                identity,
                RFGSM(adv_model),
                identity,
                identity
            ])
        else:
            attack = identity

        x_input, y_target = x_input.to(training_device), y_target.to(training_device)
        x_input = data_augmentation(x_input)

        optimiser.zero_grad()

        y_pred = model(x_input)
        loss = criterion(y_pred, y_target)

        if adv_training and attack is not None:
            x_adv = attack(x_input, y_target)
            adv_loss = criterion(model(x_adv), y_target)
            loss = 0.1 * adv_loss + 0.9 * loss

        loss.backward()
        optimiser.step()

        if not i % 50:
            # Compute relevant metrics
            y_pred_max = torch.argmax(y_pred, dim=1)  # Get the labels with the highest output probability
            correct = torch.sum(torch.eq(y_pred_max, y_target)).item()  # Count how many are equal to the true labels

            print(f'b-{i}, train accuracy: {correct / len(x_input):.3f}-loss: {loss}. attack: {type(attack).__name__}')


def train_suit(model, num_epochs, training_data_loader, data_augmentation, optimiser, criterion,
               use_cuda=True, save_model=False, path_to_file='.',
               adv_training=False, ensemble_list=None, lower_bound_ep=0.05, upper_bound_ep=0.40,
               testing_data_loader=None, testing_length=None, test_aug=None):
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    model = model.to(device)

    try:
        (model_dict, optim_dict, stored_acc) = torch.load(path_to_file, device)
        model.load_state_dict(model_dict)
        print('Old Model State Loaded!')
        optimiser.load_state_dict(optim_dict)
        print('Old Optim State Loaded!')
        best_so_far = stored_acc
    except (FileNotFoundError, IOError) as e:
        print('Saved model not found!')
        best_so_far = 0

    print(f'Starting Training -> best_so_far = {best_so_far}')
    for epoch_num in range(num_epochs):
        print(f'Starting epoch: {epoch_num}')

        model.train()
        epoch_train(model, training_data_loader, data_augmentation, optimiser, criterion,
                    training_device=device, adv_training=adv_training, ensemble_list=ensemble_list,
                    l_ep=lower_bound_ep, u_ep=upper_bound_ep)

        if testing_length is not None and testing_data_loader is not None:

            if test_aug is None:
                test_aug = identity

            model.eval()  # Put the network in eval mode
            with torch.no_grad():
                correct_total = 0

                for i, (x_batch, y_batch) in enumerate(testing_data_loader):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    x_batch = test_aug(x_batch)

                    y_pred = model(x_batch)
                    y_pred_max = torch.argmax(y_pred, dim=1)

                    correct_total += torch.sum(torch.eq(y_pred_max, y_batch)).item()

                accuracy = correct_total / testing_length
                print(f'Accuracy on the test set: {accuracy:.3f}')

                if save_model and (accuracy >= best_so_far):
                    best_so_far = accuracy
                    torch.save((model.state_dict(), optimiser.state_dict(), best_so_far), path_to_file)
                    print(f'Model saved! -> best_so_far = {best_so_far}')
                else:
                    print(f'current / best_so_far = {accuracy} / {best_so_far}')
        else:
            if save_model:
                torch.save((model.state_dict(), optimiser.state_dict(), best_so_far), path_to_file)


def robust_test_suit(model, path_to_file, testing_data_loader, testing_length, test_aug=None, attacks=None,
                     use_cuda=True):
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    if test_aug is None:
        test_aug = identity

    model = model.to(device)

    try:
        (model_dict, _, _) = torch.load(path_to_file, device)
        model.load_state_dict(model_dict)
        print('Old Model State Loaded!')
    except (FileNotFoundError, IOError) as e:
        print('Saved model not found!')
        raise ModuleNotFoundError

    correct_total = 0
    model.eval()

    if not attacks:
        attacks = [MIFGSM(model), FGSM(model), RFGSM(model)]

    for attack in attacks:
        attack_name = type(attack).__name__
        print(f'Performing Attack: {attack_name}')
        for i, (x_batch, y_batch) in enumerate(testing_data_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = test_aug(x_batch)

            x_batch_adv = attack(x_batch, y_batch)

            y_pred = model(x_batch_adv)
            y_pred_max = torch.argmax(y_pred, dim=1)

            correct_total += torch.sum(torch.eq(y_pred_max, y_batch)).item()

        accuracy = correct_total / testing_length
        print(f'{attack_name}:ep:{attack.eps} Accuracy on the test set: {accuracy}')
