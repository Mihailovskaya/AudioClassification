import sys 
sys.path.insert(0,'/content/AudioClassfication')
from utils.helper_funcs import collate_fn
import numpy as np
from modules.soundnet import SoundNetRaw as SoundNet
import torch 
import time
import torch.nn.functional as F
import os
from utils.helper_funcs import add_weight_decay
import shutil
import torchaudio
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(net, loss, best_loss, acc, best_acc, root):
    if acc > best_acc:
        best_acc = acc
        best_loss = loss
        torch.save (net, os.path.join(root,'net.pkl')) 
        print(best_acc, 'saved')

    elif acc == best_acc:
        if loss < best_loss:
            best_loss = loss
            best_acc = acc
            best_loss = loss
            torch.save (net, os.path.join(root,'net.pkl')) 
            print(best_acc, 'saved')
    return best_acc, best_loss


def train_net(net, train_loader, val_loader, run_name = '1', n_epochs = 20, log_interval = 60, save_interval = 60, loss_type='label_smooth', load_root=None, max_lr = 3e-4, n_classes=10):
   
  if not os.path.exists('/content/result'):
    os.mkdir('/content/result')
  save_path = '/content/result'
  #####################
  # optimizer         #
  #####################
  #распад веса
  wd=1e-5
  parameters = add_weight_decay(net, wd)
  #else:
    #   parameters = net.parameters()

  opt = torch.optim.AdamW(parameters,
                          lr=max_lr,
                          betas=[0.9, 0.99],
                          weight_decay=0,
                          eps=1e-8)

  lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                      max_lr=max_lr,
                                                      steps_per_epoch=len(train_loader),
                                                      epochs=n_epochs,
                                                      pct_start=0.1,
                                                      )

  #####################
  # losses            #
  #####################
  if loss_type == "label_smooth":
      from modules.losses import LabelSmoothCrossEntropyLoss
      criterion = LabelSmoothCrossEntropyLoss(smoothing=0.1, reduction='sum').to(device)

  elif loss_type == "cross_entropy":
      criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

  elif loss_type == "focal":
      from modules.losses import FocalLoss
      criterion = FocalLoss().to(device)

  elif loss_type == 'bce':
      criterion = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device)

  ####################################
  # create folder #
  ####################################
  root = os.path.join(save_path,run_name)
  if os.path.exists(root):
    shutil.rmtree(root)
    os.mkdir(root)
  else:
    os.mkdir(root)
  #####################
  # resume training   #
  #####################
  
  if load_root and load_root.exists():
      net = torch.load(load_root)

      print('checkpoints loaded')
  # enable cudnn autotuner to speed up training
  torch.backends.cudnn.benchmark = True

  #####################
  # create files  #
  #####################
  if os.path.exists(os.path.join(root, 'costs.txt')):
    my_file = open(os.path.join(root, 'costs.txt'), "w+")
    my_file.close()
  if os.path.exists(os.path.join(root, 'costs_test.txt')):
    my_file = open(os.path.join(root, 'costs_test.txt'), "w+")
    my_file.close()
  
  #####################
  # main    #
  #####################
  best_acc = -1
  best_loss = 999
  steps = 0
  acc =0
  loss = 0

  for epoch in range(1, n_epochs + 1):
      t_epoch = time.time()
      for iterno, (x, y) in enumerate(train_loader):
          t_batch = time.time()
          x = x.to(device)
          y = torch.from_numpy(np.asarray(y))
          y = y.to(device)
          pred = net(x)
          loss_cls = criterion(pred, y)

          ###################
          # Train Generator #
          ###################
          net.zero_grad()
          loss_cls.backward()
          opt.step()
          lr_scheduler.step()
          
          _, y_pred = torch.max(pred, 1)
          loss += F.cross_entropy(pred, y)
          acc += accuracy_score(y.tolist(), y_pred.tolist())
          ######################
          # metrics and save #
          ######################
          if steps % log_interval == 0: 
              print(f'{acc} interval: {log_interval}')
              acc = acc/log_interval
              loss = loss/log_interval
              file = open(os.path.join(root, "costs.txt"), "a")
              file.write(f'{steps}-{acc}-{loss}-{opt.param_groups[0]["lr"]}\n')
              file.close()     
              t_batch = time.time() - t_batch
              print(f"epoch {epoch}/{n_epochs} | iters {iterno}/{len(train_loader)} | acc: {acc:.2f} | loss: {loss:.2f} | ms/batch {(1000 * t_batch / log_interval):5.2f} ")
              start = time.time()
              acc = 0
              loss = 0
          
          if steps % save_interval == 0:
              ''' validate'''
              net.eval()
              st = time.time()
              loss_test = 0
              acc_test = 0
              cm = np.zeros((n_classes, n_classes), dtype=np.int32)
              idx_start = 0
              with torch.no_grad():
                  for i, (x, y) in enumerate(val_loader):
                      x = x.to(device)
                      y = torch.from_numpy(np.asarray(y))
                      y = y.to(device)
                      pred = net(x)
                      _, y_est = torch.max(pred, 1)
                      acc_test += accuracy_score(y.tolist(), y_est.tolist())
                      #print(acc_test)
                      
                      loss_test += F.cross_entropy(pred, y)
                      '''for t, p in zip(y.view(-1), y_est.view(-1)):
                          cm[t.long(), p.long()] += 1'''
                  loss_test /= len(val_loader)
                  acc_test /= len(val_loader)
                  #acc_cm = np.diag(cm).sum()/ len(val_loader.dataset)
              
              file = open(os.path.join(root, "costs_test.txt"), "a")
              file.write(f'{steps}-{acc_test}-{loss_test.item()}\n')
              file.close()
              
              best_acc, best_loss = save_model(net, loss_test, best_loss, acc_test, best_acc, root)

              print(f'test: Epoch {epoch} | Iters {iterno} | acc: {acc_test:.4f} loss: {loss_test:.2f} | ms/batch {1000 * (time.time() - start) / log_interval:5.2f} ')

              print("-" * 100)
              net.train()
          steps += 1

      t_epoch = time.time() - t_epoch
      print("epoch {}/{} time {:.2f}".format(epoch, n_epochs, t_epoch / log_interval))


# предобработка аудио
def preprocessing(path):
    sampling_rate = 22050
    segment_length = 114688
    
    audio, rate = torchaudio.load(path)
    sample = torchaudio.transforms.Resample(rate, sampling_rate)(audio)
    audio = sample.unsqueeze(0)
    audio.squeeze_()
    audio = 0.95 * (audio / audio.__abs__().max()).float()
    if audio.shape[0] >= segment_length:
        max_audio_start = audio.size(0) - segment_length
        audio_start = random.randint(0, max_audio_start)
        audio = audio[audio_start: audio_start + segment_length]
    else:
        audio = F.pad(audio, (0, segment_length - audio.size(0)), "constant").data

    return audio.unsqueeze(0).unsqueeze(0)

def inference(path_sound, path_net):
    net = torch.load(path_net)
    net.to(device)
    preprocessed_audio = preprocessing(path_sound)

    with torch.no_grad():
        preprocessed_audio = preprocessed_audio.to(device)
        pred = net(preprocessed_audio)
    max_elements, max_idxs = torch.max(pred, dim = 1)
    cls = int(max_idxs)
    print(f'cls: cls')


def test(path_net, data_loader, n_classes =10):
    net = torch.load(path_net)
    net.to(device)
    labels = torch.zeros(len(data_loader.dataset), dtype=torch.float32).float()
    preds = torch.zeros(len(data_loader.dataset), n_classes, dtype=torch.float32).float()
    confusion_matrix = torch.zeros(n_classes, n_classes, dtype=torch.int)
    idx_start = 0

    for i, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            pred = net(x)
            
            _, y_est = torch.max(pred, 1)
            idx_end = idx_start + y.shape[0]
            preds[idx_start:idx_end, :] = pred
            labels[idx_start:idx_end] = y
            for t, p in zip(y.view(-1), y_est.view(-1)):
              confusion_matrix[t.long(), p.long()] += 1
            print(f"{i}/{len(data_loader)}")
        idx_start = idx_end
    #acc_av = accuracy(preds.detach(), labels.detach(), [1, ])[0]
    _, preds = torch.max(preds, 1)

    acc_av= accuracy_score(labels.tolist(), preds.tolist())
    classes = ['air_conditioner','car_horn', 'children_playing', 'dog_bark', 
               'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    plot_confusion_matrix(labels.tolist(), preds.tolist(), classes)
    report = classification_report(labels.tolist(), preds.tolist(), target_names=classes)
    print(report)
    res = {
        "acc": acc_av,
        "preds": preds,
        "labels": labels.view(-1),
        "confusion_matrix": confusion_matrix
    }
    print(f'conf matrix: ')
    print(f'{confusion_matrix}')
    print("acc:{}".format(np.round(acc_av*100)/100))
    print('***************************************')
    bad_labels = []
    for i, c in enumerate(confusion_matrix):
        i_est = c.argmax(-1)
        #print(f'i: {i}, iest: {i_est}')
        #print(data_set.labels)
        if i != i_est:
            print('{} {} {}-->{}'.format(i, i_est.item(), data_set.labels[i], data_set.labels[i_est]))
            bad_labels.append([i, i_est])
    print(bad_labels)


def plot_confusion_matrix(y_true,
                          y_pred,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be pplied by setting normalize=True.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm / cm.astype(np.float).sum(axis=1)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig('saved_figure.png')
    
    image = cv2.imread("saved_figure.png")
    return image
