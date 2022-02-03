import random
import time
from fractions import Fraction

import torch
from torchattacks import FGSM, RFGSM, MIFGSM, PGD


def identity(woof, *args, **kwargs):
    return woof


def epoch_train(model, training_data_loader, data_augmentation, optimiser, criterion, training_device,
                adv_training, ensemble_list, u_ep, best_acc_seen):
    epsilon_scaling = best_acc_seen
    for i, (x_input, y_target) in enumerate(training_data_loader):

        if adv_training:
            if ensemble_list:
                adv_model = random.choice(ensemble_list)
            else:
                adv_model = model
            steps = 5
            epsilon = epsilon_scaling * u_ep * random.random()
            attack = random.choice([
                MIFGSM(adv_model, epsilon, alpha=epsilon / (steps - 1), steps=steps), identity, identity,
                FGSM(adv_model, epsilon), identity, identity,
                RFGSM(adv_model, epsilon), identity, identity,
                PGD(adv_model, epsilon, alpha=epsilon / (steps - 1), steps=steps), identity, identity,
            ])
        else:
            attack = None

        x_input, y_target = x_input.to(training_device), y_target.to(training_device)
        x_input = data_augmentation(x_input)

        optimiser.zero_grad()

        if adv_training:
            x_adv = attack(x_input, y_target)
            y_pred = model(x_adv)
            del x_adv
        else:
            y_pred = model(x_input)
        loss = criterion(y_pred, y_target)

        loss.backward()
        optimiser.step()

        if not i % 25:
            # Compute relevant metrics
            y_pred_max = torch.argmax(y_pred, dim=1)  # Get the labels with the highest output probability
            correct = float(torch.sum(torch.eq(y_pred_max, y_target)).item())  # Count num equal to the true label
            train_acc = correct / len(x_input)

            if attack is identity:
                attack_name = identity.__name__
            elif attack is None:
                attack_name = 'None'
            else:
                epsilon_scaling = train_acc
                attack_name = f'{type(attack).__name__}-ep:{attack.eps}'

            print(f'b-{i}'.ljust(10),
                  f'train accuracy: {train_acc}'.ljust(30),
                  f'loss: {loss:.4f}'.ljust(15),
                  f'attack: {attack_name}'.ljust(30))
            del correct, y_pred_max, attack_name

        del x_input, y_target, y_pred, loss


def train_suit(model, num_epochs, training_data_loader, data_augmentation, optimiser, criterion,
               use_cuda=True, save_model=False, path_to_file='.',
               adv_training=False, ensemble_list=None, upper_bound=64 / 255,
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
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        print(f'Starting epoch: {epoch_num}')

        model.train()
        epoch_train(model, training_data_loader, data_augmentation, optimiser, criterion,
                    training_device=device, adv_training=adv_training, ensemble_list=ensemble_list,
                    u_ep=min([((epoch_num + 2)/num_epochs) * upper_bound, upper_bound]), best_acc_seen=best_so_far)

        if testing_length is not None and testing_data_loader is not None:

            if test_aug is None:
                test_aug = identity

            model.eval()  # Put the network in eval mode
            with torch.no_grad():
                correct_total = 0

                for x_batch, y_batch in testing_data_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    x_batch = test_aug(x_batch)

                    y_pred = model(x_batch)
                    y_pred_max = torch.argmax(y_pred, dim=1)

                    correct_total += float(torch.sum(torch.eq(y_pred_max, y_batch)).item())
                    del x_batch, y_batch, y_pred, y_pred_max

                accuracy = correct_total / testing_length
                print(f'Accuracy on the test set: {accuracy:.3f}')

                if save_model and (accuracy >= best_so_far):
                    best_so_far = accuracy
                    torch.save((model.state_dict(), optimiser.state_dict(), best_so_far), path_to_file)
                    print(f'Model saved! -> best_so_far = {best_so_far}')
                else:
                    print(f'current / best_so_far = {accuracy} / {best_so_far}')
                del accuracy, correct_total
        else:
            if save_model:
                torch.save((model.state_dict(), optimiser.state_dict(), best_so_far), path_to_file)

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f'Training took: {duration:.3f}s'.ljust(30), f'{duration / num_epochs} s/epoch')


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
        (model_dict, _, clean_acc) = torch.load(path_to_file, device)
        model.load_state_dict(model_dict)
        print('Old Model State Loaded!', path_to_file)
    except (FileNotFoundError, IOError) as e:
        print('Saved model not found!')
        raise ModuleNotFoundError

    model.eval()

    if not attacks:
        attacks = [
            FGSM(model, 4 / 255), FGSM(model, 8 / 255), FGSM(model, 16 / 255), FGSM(model, 32 / 255),
            FGSM(model, 64 / 255),
            MIFGSM(model, 4 / 255, 1 / 255, 5), MIFGSM(model, 8 / 255, 2 / 255, 5), MIFGSM(model, 16 / 255, 4 / 255, 5),
            MIFGSM(model, 32 / 255, 8 / 255, 5), MIFGSM(model, 64 / 255, 16 / 255, 5),
            PGD(model, 4 / 255, 1 / 255, 5), PGD(model, 8 / 255, 2 / 255, 5), PGD(model, 16 / 255, 4 / 255, 5),
            PGD(model, 32 / 255, 8 / 255, 5), PGD(model, 64 / 255, 16 / 255, 5),
        ]

    print(f'Accuracy on clean test set: {clean_acc}')
    for attack in attacks:
        correct_total = 0

        attack_name = type(attack).__name__
        print(f'Performing Attack: {attack_name}')
        for i, (x_batch, y_batch) in enumerate(testing_data_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = test_aug(x_batch)

            x_batch_adv = attack(x_batch, y_batch)

            y_pred = model(x_batch_adv)
            y_pred_max = torch.argmax(y_pred, dim=1)

            correct_total += float(torch.sum(torch.eq(y_pred_max, y_batch)).item())
            del x_batch, y_batch, x_batch_adv, y_pred, y_pred_max

        accuracy = correct_total / testing_length
        print(f'{attack_name}'.ljust(10),
              f'ep:{Fraction(attack.eps).limit_denominator()}'.ljust(15),
              f'Accuracy on the test set: {accuracy:.4f}'.ljust(40))

        del accuracy, correct_total, attack_name

    del clean_acc, model
