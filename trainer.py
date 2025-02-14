import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import time


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, scaler, n_epochs, cuda, log_interval,
        metrics=[], start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    # for epoch in range(0, start_epoch):
    #     scheduler.step()

    best_precision = float("-inf")

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        start = time.time()
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, scaler)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        print(f"epoch took {time.time() - start:.4f}s")

        start = time.time()
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics,
                                       use_amp=scaler.is_enabled())
        val_loss /= len(val_loader)
        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        print(f"val took {time.time() - start:.4f}s")
        print(message)

        torch.save({
            'epoch': epoch,
            'model_name': model.default_cfg["architecture"],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, f"./last.pt")

        precisions, recalls = eval_model(val_loader.dataset, model, cuda, val_loader.batch_sampler.batch_size,
                                         use_amp=scaler.is_enabled())
        print_precision_and_recall(precisions, recalls)
        if precisions[0] > best_precision:
            print("New best")
            best_precision = precisions[0]
            torch.save(model.state_dict(), f"./best.pt")

        print("")


def eval_model(dataset, model, cuda, inference_batch_size, use_amp=False):
    print("Eval: Calculating recall and precision ...")
    with torch.no_grad():
        model.eval()

        start = time.time()
        embeddings, labels = get_embeddings(model, dataset, inference_batch_size, cuda=cuda, use_amp=use_amp)
        print(f"Embeddings: {time.time() - start:.4f}s")

        start = time.time()
        similarity_matrix = get_similarity_matrix(embeddings)
        print(f"Matrix: {time.time() - start:.4f}s")

        start = time.time()
        precisions, recalls = calculate_precision_and_recall(similarity_matrix, embeddings, labels)
        print(f"Metrics: {time.time() - start:.4f}s")

        return precisions, recalls


def calculate_precision_and_recall(similarity_matrix, embeddings, labels):
    # calculate precision@k and recall@k for k=1 to 5.
    length = len(embeddings)
    k_max = 5
    k_vals = np.arange(1, k_max + 1)
    precision_correct = np.zeros(k_max)
    recall_scores = np.zeros(k_max)
    for target_i in range(0, length):
        target_label = labels[target_i]

        # get the images with the highest cosine similarity.
        # the first result is filtered out because it's just the query image again, with a similarity of 1.
        similar_idx = np.argsort(similarity_matrix[target_i])[-(k_max + 1):-1][::-1]
        similar = [(similarity_matrix[target_i][x], labels[x]) for x in similar_idx]

        # precision
        for result_i in range(0, k_max):
            if similar[result_i][1] == target_label:
                for k in range(result_i, k_max):
                    precision_correct[k] += 1

        # recall
        similar_labels = [x[1] for x in similar]
        for k in range(0, k_max):
            if target_label in similar_labels[:k + 1]:
                recall_scores[k] += 1
    precisions = precision_correct / (k_vals * length)  # not sure if this is correct?
    recalls = recall_scores / length
    return precisions, recalls


def get_embeddings(model, dataset, inference_batch_size, cuda=True, use_amp=False):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataset), model.num_features))
        labels = np.zeros(len(dataset))
        k = 0
        loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False)
        for images, targets in loader:
            if cuda:
                images = images.cuda()
            with torch.cuda.amp.autocast(enabled=use_amp):
                embeddings[k:k + len(images)] = model(images).squeeze().data.cpu().numpy()
            labels[k:k + len(images)] = targets.numpy()
            k += len(images)
        return embeddings, labels


def get_similarity_matrix(embeddings):
    return cosine_similarity(embeddings, embeddings)


def print_precision_and_recall(precisions, recalls):
    precision_msg = "Eval: "
    recall_msg = "Eval: "
    for i in range(0, len(precisions)):
        precision_msg += f"P@{i+1}: {precisions[i]:.2%}  "
        recall_msg += f"R@{i+1}: {recalls[i]:.2%}  "
    print(precision_msg)
    print(recall_msg)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, scaler):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = torch.tensor(target, dtype=torch.int) if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(*data)
            outputs = torch.squeeze(outputs)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

        losses.append(loss.item())
        total_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics, use_amp=False):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0

        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(*data)
            outputs = torch.squeeze(outputs)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
