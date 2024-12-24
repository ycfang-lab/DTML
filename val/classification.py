import torch
from tqdm import tqdm
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score


def generate_image_classification_test(loader, G, C1, C2=None, num_classes=241):
    """
    args:
        loader: test loader
        G: Generator for generating another domain image
        C1: Classifier for test domain
        C2: Classifier for another domain
    """
    # If there is no C2, it defaults to two classifiers being the same
    C2 = C1 if C2 is None else C2
    # Testing has not yet started
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        G.eval()
        C1.eval()
        C2.eval()
        for i in tqdm(range(len(loader)), desc="Evaluating"):
            inputs, labels = iter_test.next()
            # real samples
            real = inputs.cuda()
            # Labels
            labels = labels.cuda()
            # Generate fake samples
            fake = G(real)
            # Classification results of real samples
            _, real_outputs = C1(real)
            # Classification results of fake samples
            _, fake_outputs = C2(fake)
            # The first batch of test sets
            if start_test:
                # Record real results
                all_real_output = real_outputs.float()
                # Record fake results
                all_fake_output = fake_outputs.float()
                # Record labels
                all_label = labels.float()
                # The first batch of test sets has been completed
                start_test = False
            # Subsequent test sets
            else:
                # Splicing real results
                all_real_output = torch.cat((all_real_output, real_outputs.float()), 0)
                # Splicing fake results
                all_fake_output = torch.cat((all_fake_output, fake_outputs.float()), 0)
                # Splicing labels
                all_label = torch.cat((all_label, labels), 0)

    # real prediction label
    _, real_predict = torch.max(all_real_output, 1)
    # fake prediction label
    _, fake_predict = torch.max(all_fake_output, 1)

    # acc
    accuracy = Accuracy(
        task="multiclass", average="micro", num_classes=num_classes
    ).cuda()
    precision = Precision(
        task="multiclass", average="macro", num_classes=num_classes
    ).cuda()
    recall = Recall(
        task="multiclass", average="macro", num_classes=num_classes
    ).cuda()
    f1score = F1Score(
        task="multiclass", average="macro", num_classes=num_classes
    ).cuda()
    # real samples
    accuracy.update(real_predict, all_label)
    precision.update(real_predict, all_label)
    recall.update(real_predict, all_label)
    f1score.update(real_predict, all_label)
    real_accuracy = accuracy.compute().item()  
    real_precision = precision.compute().item()  
    real_recall = recall.compute().item()  
    real_f1score = f1score.compute().item()  
    # fake samples
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1score.reset()
    accuracy.update(fake_predict, all_label)
    precision.update(fake_predict, all_label)
    recall.update(fake_predict, all_label)
    f1score.update(fake_predict, all_label)
    fake_accuracy = accuracy.compute().item()  
    fake_precision = precision.compute().item()  
    fake_recall = recall.compute().item()  
    fake_f1score = f1score.compute().item()  
    accuracy = max(real_accuracy, fake_accuracy)
    precision = max(real_precision, fake_precision)
    recall = max(real_recall, fake_recall)
    f1score = max(real_f1score, fake_f1score)

    return (
        (accuracy, real_accuracy, fake_accuracy),
        (precision, real_precision, fake_precision),
        (recall, real_recall, fake_recall),
        (f1score, real_f1score, fake_f1score)
    )