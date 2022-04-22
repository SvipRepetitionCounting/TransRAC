""" test of TransRAC """
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.my_tools import paint_smi_matrixs, plot_inference, density_map

torch.manual_seed(1)


def test_loop(n_epochs, model, test_set, inference=True, batch_size=1, lastckpt=None, paint=False, device_ids=[0]):
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    currEpoch = 0
    testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=10)
    model = nn.DataParallel(model.to(device), device_ids=device_ids)

    if lastckpt is not None:
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        del checkpoint

    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        testOBO = []
        testMAE = []
        predCount = []
        Count = []
        if inference:
            with torch.no_grad():
                batch_idx = 0
                pbar = tqdm(testloader, total=len(testloader))
                for input, target in pbar:
                    model.eval()
                    acc = 0
                    input = input.to(device)
                    count = torch.sum(target, dim=1).round().to(device)
                    output, sim_matrix = model(input)
                    predict_count = torch.sum(output, dim=1).round()

                    mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                          predict_count.flatten().shape[0]  # mae

                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    OBO = acc / predict_count.flatten().shape[0]
                    testOBO.append(OBO)
                    MAE = mae.item()
                    testMAE.append(MAE)

                    predCount.append(predict_count.item())
                    Count.append(count.item())
                    print('predict count :{0}, groundtruth :{1}'.format(predict_count.item(), count.item()))
                    batch_idx += 1

        print("MAE:{0},OBO:{1}".format(np.mean(testMAE), np.mean(testOBO)))
        # plot_inference(predict_count, count)
