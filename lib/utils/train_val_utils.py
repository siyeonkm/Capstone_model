import torch
from torch.nn import functional as F
from tqdm import tqdm
n_iters = 0

def train_fol_ego(epoch, args, fol_model, ego_pred_model, optimizer, train_gen, verbose=True):
    fol_model.train()
    ego_pred_model.train()

    avg_fol_loss = 0
    avg_ego_pred_loss = 0

    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):
            input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion = data

            # forward
            ego_predictions = ego_pred_model(input_ego_motion)
            fol_predictions = fol_model(input_bbox, input_flow, ego_predictions)

            # compute loss
            fol_loss = rmse_loss_fol(fol_predictions, target_bbox)
            ego_pred_loss = rmse_loss_fol(ego_predictions, target_ego_motion)
            loss_to_optimize = args.lambda_fol * fol_loss +  args.lambda_ego * ego_pred_loss

            avg_fol_loss += fol_loss.item()
            avg_ego_pred_loss += ego_pred_loss.item()

            # optimize
            optimizer.zero_grad()
            loss_to_optimize.backward()
            optimizer.step()

            if verbose and batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t FOL loss: {:.4f}\t Ego pred loss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_gen.dataset),
                    100. * batch_idx / len(train_gen), fol_loss.item()/input_bbox.shape[0], ego_pred_loss.item()/input_ego_motion.shape[0]))
        avg_fol_loss /= len(train_gen.dataset)
        avg_ego_pred_loss /= len(train_gen.dataset)
        avg_train_loss = avg_fol_loss + avg_ego_pred_loss
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t FOL loss: {:.4f}\t Ego pred loss: {:.4f} Total: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_gen.dataset),
                    100. * batch_idx / len(train_gen), avg_fol_loss, avg_ego_pred_loss, avg_train_loss))


    return avg_train_loss, avg_fol_loss, avg_ego_pred_loss

def val_fol_ego(epoch, args, fol_model, ego_pred_model, val_gen, verbose=True):
    fol_model.eval()
    ego_pred_model.eval()

    fol_loss = 0
    ego_pred_loss = 0
    loader = tqdm(val_gen, total=len(val_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion = data

            # forward
            ego_predictions = ego_pred_model(input_ego_motion)
            fol_predictions = fol_model(input_bbox, input_flow, ego_predictions)

            # compute loss
            fol_loss += rmse_loss_fol(fol_predictions, target_bbox).item()
            ego_pred_loss += rmse_loss_fol(ego_predictions, target_ego_motion).item()

    fol_loss /= len(val_gen.dataset)
    ego_pred_loss /= len(val_gen.dataset)
    avg_val_loss = fol_loss + ego_pred_loss
    if verbose:
        print('\nVal set: Average FOL loss: {:.4f}, Average Ego pred loss: {:.4f}, Total: {:.4f}\n'.format(fol_loss, ego_pred_loss, avg_val_loss))

    return avg_val_loss, fol_loss, ego_pred_loss


def rmse_loss_fol(x_pred, x_true):
    L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=3))
    L2_all_pred = torch.sum(L2_diff, dim=2)

    L2_mean_pred = torch.mean(L2_all_pred, dim=1)
    L2_mean_pred = torch.mean(L2_mean_pred, dim=0)

    return L2_mean_pred
