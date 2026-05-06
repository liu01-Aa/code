# import numpy as np
# import torch
# import deepcoloring as dc
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# def train(train_dataloder, model, max_n_objects=20, niter=300, lr=1e-3):
#
#
#     print("model.train():  ")
#
#     model.train()
#
#     mask_builder = dc.Discriminative_binary_Mask(max_n_objects=max_n_objects, sem_class=2)
#     criterion_disc = dc.DiscriminativeLoss(delta_var=0.5,
#                                            delta_dist=1.5,
#                                            norm=2,
#                                            usegpu=True).to(device)
#     criterion_ce = torch.nn.CrossEntropyLoss().to(device)
#     criterion_mse = torch.nn.MSELoss().to(device) #130
#     #
#
#     # parameters = model.parameters()
#     # optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.001)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=10, verbose=True)
#     best_loss = np.inf
#
#     for epoch in range(niter):
#         print(f'epoch: {epoch}')
#         disc_losses = []
#         ce_losses = []
#         mse_losses = []
#         for batch in train_dataloder:
#             images, lables = batch
#             images = images.to(device)
#             sem_predict, ins_predict, objects_pre = model(images)
#             # print(ins_predict.shape)
#             # print(sem_predict.shape)
#             # print(objects_pre.shape)
#             # exit()
#
#             ins_mask, sem_mask, objects = mask_builder(lables)
#             loss = 0
#             # discriminative
#             disc_loss = criterion_disc(ins_predict, ins_mask, objects)
#
#             loss = loss + disc_loss
#             disc_losses.append(disc_loss.data.cpu().numpy())
#             _, sem_labels_ce = sem_mask.max(1)
#             ce_loss = criterion_ce(sem_predict.permute(0, 2, 3, 1).contiguous().view(-1, 2), sem_labels_ce.view(-1))
#             loss = loss + ce_loss
#             ce_losses.append(ce_loss.data.cpu().numpy())
#
#             # Huber loss
#             objects = objects.unsqueeze(dim=1)
#             gpu_object_norm = objects.float() / max_n_objects
#             mse_loss = criterion_mse(objects_pre, gpu_object_norm)
#
#             loss = loss + mse_loss
#             mse_losses.append(mse_loss.data.cpu().numpy())
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#
#         disc_loss = np.mean(disc_losses)
#         ce_loss = np.mean(ce_losses)
#         mse_loss = np.mean(mse_losses)
#
#         print(f'DiscriminativeLoss: {disc_loss:.4f}')
#         print(f'CrossEntropyLoss: {ce_loss:.4f}')
#         print(f'MSELoss:{mse_loss:.4f}')
#
#         scheduler.step(disc_loss)
#         if disc_loss < best_loss:
#             best_loss = disc_loss
#             modelname = 'RSUnet01b.th'
#             torch.save(model.state_dict(), modelname)
#     return model
#-------------------以下为修改-------------------------
import numpy as np
import torch
import deepcoloring as dc
from torch.cuda.amp import GradScaler, autocast  # 新增AMP相关导入

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train文件 将lr从1e-3改为3e-4
def train(train_dataloder, model, max_n_objects=20, niter=300, lr=1e-3):
    print("model.train(): ")
    model.train()
#1----------------------

#----------------------1
    # 初始化梯度缩放器（AMP）
    scaler = GradScaler()

    # 梯度累积步数（根据GPU内存调整，建议4-8）
    accumulation_steps = 4

    mask_builder = dc.Discriminative_binary_Mask(max_n_objects=max_n_objects, sem_class=2)
    criterion_disc = dc.DiscriminativeLoss(delta_var=0.5,
                                           delta_dist=1.5,
                                           norm=2,
                                           usegpu=True).to(device)
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)
    criterion_mse = torch.nn.MSELoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=10,
                                                           verbose=True)
    best_loss = np.inf

    for epoch in range(niter):
        print(f'epoch: {epoch}')
        disc_losses = []
        ce_losses = []
        mse_losses = []

        optimizer.zero_grad()  # 移到epoch循环开始处

        for batch_idx, batch in enumerate(train_dataloder):
            images, lables = batch

            images = images.to(device, non_blocking=True)

            # 使用混合精度前向传播
            with autocast():
                sem_predict, ins_predict, objects_pre = model(images)

                ins_mask, sem_mask, objects = mask_builder(lables)
                # # ===== 新增调试代码 =====
                # print("instance_mask 类型:", type( ins_mask))
                # if isinstance( ins_mask, (list, tuple)):
                #     print("子元素形状:", [x.shape if hasattr(x, 'shape') else type(x) for x in  ins_mask])
                # elif isinstance( ins_mask, torch.Tensor):
                #     print("整体形状:",  ins_mask.shape)
                # # ======================
                # 计算各损失（自动保持适当精度）
                loss = 0
                disc_loss = criterion_disc(ins_predict, ins_mask, objects)
                loss += disc_loss / accumulation_steps  # 损失归一化

                _, sem_labels_ce = sem_mask.max(1)
                ce_loss = criterion_ce(sem_predict.permute(0, 2, 3, 1).contiguous().view(-1, 2),
                                       sem_labels_ce.view(-1))
                loss += ce_loss / accumulation_steps

                objects = objects.unsqueeze(dim=1)
                gpu_object_norm = objects.float() / max_n_objects
                mse_loss = criterion_mse(objects_pre, gpu_object_norm)
                loss += mse_loss / accumulation_steps

            # 缩放损失并反向传播（自动处理混合精度）
            scaler.scale(loss).backward()

            # 梯度累积达到指定步数时更新权重
            if (batch_idx + 1) % accumulation_steps == 0:

                # 梯度裁剪（可选）
                scaler.unscale_(optimizer)#-----------
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # 记录原始损失值（乘以累积步数还原）
                disc_losses.append(disc_loss.item() * accumulation_steps)
                ce_losses.append(ce_loss.item() * accumulation_steps)
                mse_losses.append(mse_loss.item() * accumulation_steps)

                # print(f'Batch: {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}')

        # 处理最后不足accumulation_steps的批次
        if len(train_dataloder) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 计算epoch平均损失
        disc_loss = np.mean(disc_losses) if disc_losses else 0
        ce_loss = np.mean(ce_losses) if ce_losses else 0
        mse_loss = np.mean(mse_losses) if mse_losses else 0

        print(f'DiscriminativeLoss: {disc_loss:.4f}')
        print(f'CrossEntropyLoss: {ce_loss:.4f}')
        print(f'MSELoss:{mse_loss:.4f}')

        scheduler.step(disc_loss)
        if disc_loss < best_loss:
            best_loss = disc_loss
            modelname = 'RSUnet01b.th'
            torch.save(model.state_dict(), modelname)

    return model
