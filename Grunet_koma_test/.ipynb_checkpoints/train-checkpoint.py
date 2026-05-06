import numpy as np
import torch
import deepcoloring as dc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(train_dataloder, model, max_n_objects=20, niter=300, lr=1e-3):


    print("model.train():  ")

    model.train()

    mask_builder = dc.Discriminative_binary_Mask(max_n_objects=max_n_objects, sem_class=2)
    criterion_disc = dc.DiscriminativeLoss(delta_var=0.5,
                                           delta_dist=1.5,
                                           norm=2,
                                           usegpu=True).to(device)
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)
    criterion_mse = torch.nn.MSELoss().to(device) #130
    #

    # parameters = model.parameters()
    # optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=10, verbose=True)
    best_loss = np.inf

    for epoch in range(niter):
        print(f'epoch: {epoch}')
        disc_losses = []
        ce_losses = []
        mse_losses = []
        for batch in train_dataloder:
            images, lables = batch
            images = images.to(device)
            sem_predict, ins_predict, objects_pre = model(images)
            # print(ins_predict.shape)
            # print(sem_predict.shape)
            # print(objects_pre.shape)
            # exit()

            ins_mask, sem_mask, objects = mask_builder(lables)
            loss = 0
            # discriminative
            disc_loss = criterion_disc(ins_predict, ins_mask, objects)

            loss = loss + disc_loss
            disc_losses.append(disc_loss.data.cpu().numpy())
            _, sem_labels_ce = sem_mask.max(1)
            ce_loss = criterion_ce(sem_predict.permute(0, 2, 3, 1).contiguous().view(-1, 2), sem_labels_ce.view(-1))
            loss = loss + ce_loss
            ce_losses.append(ce_loss.data.cpu().numpy())

            # Huber loss
            objects = objects.unsqueeze(dim=1)
            gpu_object_norm = objects.float() / max_n_objects
            mse_loss = criterion_mse(objects_pre, gpu_object_norm)

            loss = loss + mse_loss
            mse_losses.append(mse_loss.data.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        disc_loss = np.mean(disc_losses)
        ce_loss = np.mean(ce_losses)
        mse_loss = np.mean(mse_losses)

        print(f'DiscriminativeLoss: {disc_loss:.4f}')
        print(f'CrossEntropyLoss: {ce_loss:.4f}')
        print(f'MSELoss:{mse_loss:.4f}')

        scheduler.step(disc_loss)
        if disc_loss < best_loss:
            best_loss = disc_loss
            modelname = 'RSUnet01b.th'
            torch.save(model.state_dict(), modelname)
    return model





