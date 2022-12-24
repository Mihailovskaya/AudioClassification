# AudioClassification
Этот репозиторий предназначен для классификации звуков окружающей среды. Данные расположены в папке dataset_splitted, распределены по папкам train/test/val в соотношении 0.7/0.15/0.15 и распледелены по классам:
  0 = air_conditioner
  1 = car_horn
  2 = children_playing
  3 = dog_bark
  4 = drilling
  5 = engine_idling
  6 = gun_shot
  7 = jackhammer
  8 = siren
  9 = street_music

В ноутбуке храниться все, что нужно для запуска сети:
 1) скачивание нужных файлов с гита
 2) создание сети из файла modules/soundnet.pay
 3) создание train/val/test датасета с помощью класса из файла utils/audio_dataset.py наследующего torch.utils.data.Dataset
 4) обучение сети через функцию train_net из train.py
 5) инференс одного аудио через функцию inference из train.py
 6) тест тестового набора данных с помощью функции test из train.py
 7) вывод accuracy и loss при обучении 
 8) вывод confusion matrix 

В f'result/{run_name}' сохраняется:
        лучшая сеть net.pkl,
        costs.txt - записи о train  датасете f'{step}-{acc}-{loss}'
        costs_test.txt - записи о val датасете f'{step}-{acc}-{loss}'
