import torch
import torchaudio
import torch.nn.functional as F
import os
import glob
import random

sep = os.path.sep

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 mode,
                 segment_length,
                 sampling_rate
                ):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self._get_labels(root)
        if mode == 'train':
            fnames = self.get_training_list(root, 'train')
        elif mode == 'val':
            fnames = self.get_training_list(root, 'val')
        elif mode == 'test':
            fnames = self.get_training_list(root, 'test')
        else:
            raise ValueError
        self.audio_files = sorted(fnames)
        self.label2idx = dict(zip(self.labels, range(len(self.labels))))
        '''if self.use_background:
            self.bg_aug = glob.glob(root + f"{sep}_background_noise_{sep}*.wav")
            self.bg_aug = [f for f in self.bg_aug if 'noise' not in os.path.basename(f)]
            self.bg_aug = [torchaudio.load(f)[0][0].detach() for f in self.bg_aug]
            self.bg_aug = [x for x in self.bg_aug]'''

  

    def _get_labels(self, root):
        #f_names = glob.glob(root + f"{sep}**{sep}*.wav")
        #self.labels = sorted(list(set([f.split(f'{os.path.sep}')[-2] for f in f_names])))
        #self.labels = sorted([l for l in self.labels if l != '_background_noise_'])
        self.labels = ['0','1','2','3','4','5','6','7','8','9']

    def __getitem__(self, index):
        fname = self.audio_files[index]
        if '/' in fname:
            fname = fname.replace('/',sep)
        label = fname.split(f'{sep}')[-2]
        #print(f'label: {label}, fname: {fname}')
        label = int(label)
        #label = self.label2idx[label]
        audio, sampling_rate = torchaudio.load(fname)
        audio.squeeze_()
        audio = 0.95 * (audio / audio.__abs__().max()).float()
        

        assert ("sampling rate of the file is not as configured in dataset, will cause slow fetch {}".format(
            sampling_rate != self.sampling_rate))
        if audio.shape[0] >= self.segment_length:
            #print(f'audio.shape[0]: {audio.shape[0]}, self.segment_length: {self.segment_length}')
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start: audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data
        #print(audio.shape)
        
        return audio.unsqueeze(0), label

    def __len__(self):
        return len(self.audio_files)

    def get_training_list(self, root, mode):
        paths = []
        for cls in os.listdir(os.path.join(root,mode)):
          for file_name in os.listdir(os.path.join(root,mode,cls)):
              file_path = os.path.join(root,mode,cls,file_name)
              paths.append(file_path)

        return paths


if __name__ == "__main__":
    pass
