# coding=utf-8
"""https://github.com/google-research/google-research/blob/master/ieg/dataset_utils/datasets.py#L564"""

from math import ceil
from dataclasses import dataclass
from typing import Any

import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf

import jax
from flax import jax_utils


# CIFAR-100 superclass groupings: each list contains fine-label indices
# that belong to the same superclass. Used for asymmetric noise generation.
CIFAR100_SUPERCLASSES = {
    'aquatic_mammals':       [4, 30, 55, 72, 95],
    'fish':                  [1, 32, 67, 73, 91],
    'flowers':               [54, 62, 70, 82, 92],
    'food_containers':       [9, 10, 16, 28, 61],
    'fruit_and_vegetables':  [0, 51, 53, 57, 83],
    'household_electrical':  [22, 39, 40, 86, 87],
    'household_furniture':   [5, 20, 25, 84, 94],
    'insects':               [6, 7, 14, 18, 24],
    'large_carnivores':      [3, 42, 43, 88, 97],
    'large_man-made':        [12, 17, 37, 68, 76],
    'large_natural_scenes':  [23, 33, 49, 60, 71],
    'large_omnivores':       [15, 19, 21, 31, 38],
    'medium_mammals':        [34, 63, 64, 66, 75],
    'non-insect_inverts':    [26, 45, 77, 79, 99],
    'people':                [2, 11, 35, 46, 98],
    'reptiles':              [27, 29, 44, 78, 93],
    'small_mammals':         [36, 50, 65, 74, 80],
    'trees':                 [47, 52, 56, 59, 96],
    'vehicles_1':            [8, 13, 48, 58, 90],
    'vehicles_2':            [41, 69, 81, 85, 89],
}


def verbose_data(which_set, data, label):
  """Prints the number of data per class for a dataset."""
  text = [f'{which_set} size: {data.shape[0]}']
  text += [f'class{i}-{len(np.where(label == i)[0])}' for i in range(label.max() + 1)]
  text = ' '.join(text) + '\n'
  tf.logging.info(text)


def shuffle_dataset(data, label, others=None, class_balanced=False):
  """Shuffles the dataset with class balancing option.

  Args:
    data: A numpy 4D array.
    label: A numpy array.
    others: Optional array corresponded with data and label.
    class_balanced: If True, after shuffle, data of different classes are
      interleaved and balanced [1,2,3...,1,2,3.].

  Returns:
    Shuffled inputs.
  """
  if class_balanced:
    sorted_ids = []

    for i in range(label.max() + 1):
      tmp_ids = np.where(label == i)[0]
      np.random.shuffle(tmp_ids)
      sorted_ids.append(tmp_ids)

    sorted_ids = np.stack(sorted_ids, 0)
    sorted_ids = np.transpose(sorted_ids, axes=[1, 0])
    ids = np.reshape(sorted_ids, (-1,))

  else:
    ids = np.arange(data.shape[0])
    np.random.shuffle(ids)

  if others is None:
    return data[ids], label[ids]
  else:
    return data[ids], label[ids], others[ids]


def _build_cifar100_asymmetric_transition(noise_ratio):
  """Build a 100x100 transition matrix for CIFAR-100 asymmetric noise.

  Within each superclass, labels are flipped to the next class in a cyclic
  manner (e.g., for superclass [a, b, c, d, e]: a->b, b->c, ..., e->a)
  with probability `noise_ratio`.

  Args:
    noise_ratio: Probability of flipping a label to the next class within
      its superclass.

  Returns:
    A 100x100 numpy transition matrix.
  """
  n_classes = 100
  p = np.eye(n_classes)

  for superclass_name, members in CIFAR100_SUPERCLASSES.items():
    n_members = len(members)
    for i in range(n_members):
      src = members[i]
      dst = members[(i + 1) % n_members]  # cyclic: next class in superclass
      p[src, src] = 1.0 - noise_ratio
      p[src, dst] = noise_ratio

  return p


def load_asymmetric_cifar10(x, y, noise_ratio, n_val, random_seed=12345):
  """Create CIFAR-10 asymmetric noisy data."""

  def _generate_asymmetric_noise(y_train, n):
    """Generate cifar10 asymmetric label noise.

    Asymmetric noise confuses
      automobile <- truck
      bird -> airplane
      cat <-> dog
      deer -> horse

    Args:
      y_train: label numpy tensor
      n: noise ratio

    Returns:
      corrupted y_train.
    """
    assert y_train.max() == 10 - 1
    classes = 10
    p = np.eye(classes)

    # automobile <- truck
    p[9, 9], p[9, 1] = 1. - n, n
    # bird -> airplane
    p[2, 2], p[2, 0] = 1. - n, n
    # cat <-> dog
    p[3, 3], p[3, 5] = 1. - n, n
    p[5, 5], p[5, 3] = 1. - n, n
    # automobile -> truck
    p[4, 4], p[4, 7] = 1. - n, n
    tf.logging.info('Asymmetric corruption p:\n {}'.format(p))

    noise_y = y_train.copy()
    r = np.random.RandomState(random_seed)

    for i in range(noise_y.shape[0]):
      c = y_train[i]
      s = r.multinomial(1, p[c, :], 1)[0]
      noise_y[i] = np.where(s == 1)[0]

    actual_noise = (noise_y != y_train).mean()
    assert actual_noise > 0.0

    return noise_y

  n_img = x.shape[0]
  n_classes = 10

  # holdout balanced clean
  val_idx = []
  if n_val > 0:
    for cc in range(n_classes):
      tmp_idx = np.where(y == cc)[0]
      val_idx.append(
          np.random.choice(tmp_idx, n_val // n_classes, replace=False))
    val_idx = np.concatenate(val_idx, axis=0)

  train_idx = list(set([a for a in range(n_img)]).difference(set(val_idx)))
  if n_val > 0:
    valdata, vallabel = x[val_idx], y[val_idx]
  traindata, trainlabel = x[train_idx], y[train_idx]
  trainlabel = trainlabel.squeeze()
  label_corr_train = trainlabel.copy()

  trainlabel = _generate_asymmetric_noise(trainlabel, noise_ratio)

  if len(trainlabel.shape) == 1:
    trainlabel = np.reshape(trainlabel, [trainlabel.shape[0], 1])

  traindata, trainlabel, label_corr_train = shuffle_dataset(  # pylint: disable=unbalanced-tuple-unpacking
      traindata, trainlabel, label_corr_train)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel)


def load_asymmetric_cifar100(x, y, noise_ratio, n_val, random_seed=12345):
  """Create CIFAR-100 asymmetric noisy data.

  Noise is generated by flipping labels within superclass groups cyclically.
  For example, within 'aquatic_mammals' [beaver, dolphin, otter, seal, whale],
  beaver -> dolphin, dolphin -> otter, ..., whale -> beaver.

  Args:
    x: 4D numpy array of images.
    y: 1D/2D numpy array of labels.
    noise_ratio: Probability of flipping each label.
    n_val: Number of validation samples to holdout.
    random_seed: Random seed for reproducibility.

  Returns:
    Tuple of (traindata, trainlabel, label_corr_train), (valdata, vallabel).
  """
  n_img = x.shape[0]
  n_classes = 100

  p = _build_cifar100_asymmetric_transition(noise_ratio)
  tf.logging.info('CIFAR-100 Asymmetric corruption p (non-diagonal entries):\n')
  for superclass_name, members in CIFAR100_SUPERCLASSES.items():
    sub_p = p[np.ix_(members, members)]
    tf.logging.info(f'  {superclass_name} {members}:\n  {sub_p}\n')

  # holdout balanced clean
  val_idx = []
  if n_val > 0:
    for cc in range(n_classes):
      tmp_idx = np.where(y == cc)[0]
      val_idx.append(
          np.random.choice(tmp_idx, n_val // n_classes, replace=False))
    val_idx = np.concatenate(val_idx, axis=0)

  train_idx = list(set([a for a in range(n_img)]).difference(set(val_idx)))
  if n_val > 0:
    valdata, vallabel = x[val_idx], y[val_idx]
  traindata, trainlabel = x[train_idx], y[train_idx]
  trainlabel = trainlabel.squeeze()
  label_corr_train = trainlabel.copy()

  # Apply asymmetric noise via transition matrix
  noise_y = trainlabel.copy()
  r = np.random.RandomState(random_seed)
  for i in range(noise_y.shape[0]):
    c = trainlabel[i]
    s = r.multinomial(1, p[c, :], 1)[0]
    noise_y[i] = np.where(s == 1)[0]

  actual_noise = (noise_y != trainlabel).mean()
  tf.logging.info(f'CIFAR-100 asymmetric actual noise ratio: {actual_noise:.4f}')
  trainlabel = noise_y

  if len(trainlabel.shape) == 1:
    trainlabel = np.reshape(trainlabel, [trainlabel.shape[0], 1])

  traindata, trainlabel, label_corr_train = shuffle_dataset(  # pylint: disable=unbalanced-tuple-unpacking
      traindata, trainlabel, label_corr_train)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel)


# Keep backward-compatible alias
load_asymmetric = load_asymmetric_cifar10


def load_train_val_uniform_noise(x, y, n_classes, n_val, noise_ratio):
  """Make noisy data and holdout a clean val data.

  Constructs training and validation datasets, with controllable amount of
  noise ratio.

  Args:
    x: 4D numpy array of images
    y: 1D numpy array of labels of images
    n_classes: The number of classes.
    n_val: The number of validation data to holdout from train.
    noise_ratio: A float number that decides the random noise ratio.

  Returns:
    traindata: Train data.
    trainlabel: Train noisy label.
    label_corr_train: True clean label.
    valdata: Validation data.
    vallabel: Validation label.
  """
  n_img = x.shape[0]
  val_idx = []
  if n_val > 0:
    # Splits a clean holdout set
    for cc in range(n_classes):
      tmp_idx = np.where(y == cc)[0]
      val_idx.append(
          np.random.choice(tmp_idx, n_val // n_classes, replace=False))
    val_idx = np.concatenate(val_idx, axis=0)

  train_idx = list(set([a for a in range(n_img)]).difference(set(val_idx)))
  # split validation set
  if n_val > 0:
    valdata, vallabel = x[val_idx], y[val_idx]
  traindata, trainlabel = x[train_idx], y[train_idx]
  # Copies the true label for verification
  label_corr_train = trainlabel.copy()
  # Adds uniform noises
  mask = np.random.rand(len(trainlabel)) <= noise_ratio
  random_labels = np.random.choice(n_classes, mask.sum())
  flipped_labels = (trainlabel.squeeze() + np.random.choice(np.arange(1, n_classes), len(trainlabel))) % n_classes

  trainlabel = np.expand_dims(np.where(mask, flipped_labels, trainlabel.squeeze()), 1)

  # Shuffles dataset
  traindata, trainlabel, label_corr_train = shuffle_dataset(  # pylint: disable=unbalanced-tuple-unpacking
      traindata, trainlabel, label_corr_train)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel)


@dataclass
class TFIterWrapper:
    tfds_iter: Any
    data_info: dict
    n_data: int
    step_per_epoch: int
    
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count > len(self):
            raise StopIteration
        else:
            self.count += 1
            return next(self.tfds_iter)
        
    def __len__(self):
        return self.step_per_epoch


def inject_noise(x_train, y_train, target_ratio, seed, data_info, noise_type='uniform', data_ratio=1.0):
    dataset_name = data_info.get('dataset', 'cifar10')
    
    if noise_type == 'asymmetric':
      if dataset_name == 'cifar100':
        (x_train, y_train_noisy, y_train_true), _ = load_asymmetric_cifar100(
                                                      x_train, y_train,
                                                      random_seed=seed,
                                                      noise_ratio=target_ratio,
                                                      n_val=0)
      else:
        (x_train, y_train_noisy, y_train_true), _ = load_asymmetric_cifar10(
                                                      x_train, y_train,
                                                      random_seed=seed,
                                                      noise_ratio=target_ratio,
                                                      n_val=0)
    elif noise_type == 'uniform':
      (x_train, y_train_noisy, y_train_true), _ = load_train_val_uniform_noise(
                                                    x_train, y_train,
                                                    n_classes=data_info['num_classes'],
                                                    noise_ratio=target_ratio,
                                                    n_val=0)
    
    if data_ratio != 1.0:
      x_train, _, y_train_noisy, _ = train_test_split(x_train, y_train_noisy, test_size=int(x_train.shape[0] * (1 - data_ratio)), random_state=42, stratify=y_train_noisy)
      y_train_true = y_train_noisy
    
    conf_mat = sklearn_metrics.confusion_matrix(y_train_true, y_train_noisy)
    conf_mat = conf_mat / np.sum(conf_mat, axis=1, keepdims=True)
    print('Corrupted confusion matirx\n {}'.format(conf_mat))
    
    return x_train, y_train_noisy


def _create_ds_and_iterator(data, data_info, batch_size, is_train=True):
    """Creates a tf.data.Dataset and returns a JAX-compatible iterator.

    Args:
      data: Tuple of (images, labels).
      data_info: Dict with dataset metadata including rgb_mean, rgb_sdv.
      batch_size: Batch size.
      is_train: Whether to apply training augmentations.

    Returns:
      A prefetched JAX iterator.
    """
    input_shape = data_info['input_shape']
    h, w, c = input_shape[1], input_shape[2], input_shape[3]
    pad = 4  # standard padding for CIFAR

    ds = tf.data.Dataset.from_tensor_slices({"image": data[0], "label": data[1]}).cache()

    def decode_example(sample):
        image = tf.cast(sample['image'], tf.float32)
        image = (image - data_info['rgb_mean']) / data_info['rgb_sdv']
        if is_train:
            image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], 'CONSTANT')
            image = tf.image.random_crop(image, [h, w, c])
            image = tf.image.random_flip_left_right(image)
        batch = {'sample': image, 'target': sample['label']}
        return batch

    ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if is_train:
        ds = ds.shuffle(data[1].shape[0], seed=0, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(10)

    def _prepare(x):
        x = x._numpy()
        return x.reshape((jax.local_device_count(), -1) + x.shape[1:])

    it = map(lambda xs: jax.tree_util.tree_map(_prepare, xs), ds)
    it = jax_utils.prefetch_to_device(it, 2)
    return it


def get_cifar10_lable_noise_datasets(batch_size, target_ratio, seed, noise_type='uniform', data_ratio=1.0):
    """Creates CIFAR-10 noisy label loader as tf.data.Dataset."""
    np.random.seed(seed)

    data_info = {'dataset': 'cifar10', 'input_shape': (1, 32, 32, 3), 'num_classes': 10, 
                 'task': 'classification',
                 'rgb_mean': 255*tf.constant([0.4914, 0.4822, 0.4465], shape=[1, 1, 3], dtype=tf.float32),
                 'rgb_sdv': 255*tf.constant([0.2023, 0.1994, 0.2010], shape=[1, 1, 3], dtype=tf.float32)}
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    x_train, y_train = shuffle_dataset(x_train, y_train.astype(np.int32))
    x_train, y_train_noisy = inject_noise(x_train, y_train, target_ratio, seed, data_info, noise_type=noise_type, data_ratio=data_ratio)
    train_size = x_train.shape[0]
    train_it = _create_ds_and_iterator((x_train, y_train_noisy.squeeze()), data_info, batch_size, is_train=True)
    tf_train_it = TFIterWrapper(train_it, data_info, train_size, ceil(train_size/batch_size))
    
    x_test, y_test = shuffle_dataset(x_test, y_test.astype(np.int32))
    val_size = x_test.shape[0]
    val_it = _create_ds_and_iterator((x_test, y_test.squeeze()), data_info, batch_size, is_train=False)
    tf_val_it = TFIterWrapper(val_it, data_info, val_size, ceil(val_size/batch_size))

    verbose_data('train', x_train, y_train)
    verbose_data('test', x_test, y_test)

    return tf_train_it, tf_val_it


def get_cifar100_lable_noise_datasets(batch_size, target_ratio, seed, noise_type='uniform', data_ratio=1.0):
    """Creates CIFAR-100 noisy label loader as tf.data.Dataset.

    Supports both uniform and asymmetric (superclass-based) noise.

    Args:
      batch_size: Batch size for training/validation.
      target_ratio: Noise ratio (0.0 to 1.0).
      seed: Random seed.
      noise_type: 'uniform' or 'asymmetric'.
      data_ratio: Fraction of training data to use (1.0 = all).

    Returns:
      tf_train_it: Training iterator wrapper.
      tf_val_it: Validation iterator wrapper.
    """
    np.random.seed(seed)

    # CIFAR-100 normalization stats
    data_info = {'dataset': 'cifar100', 'input_shape': (1, 32, 32, 3), 'num_classes': 100,
                 'task': 'classification',
                 'rgb_mean': 255*tf.constant([0.5071, 0.4867, 0.4408], shape=[1, 1, 3], dtype=tf.float32),
                 'rgb_sdv': 255*tf.constant([0.2675, 0.2565, 0.2761], shape=[1, 1, 3], dtype=tf.float32)}

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    x_train, y_train = shuffle_dataset(x_train, y_train.astype(np.int32))
    x_train, y_train_noisy = inject_noise(x_train, y_train, target_ratio, seed, data_info, noise_type=noise_type, data_ratio=data_ratio)
    train_size = x_train.shape[0]
    train_it = _create_ds_and_iterator((x_train, y_train_noisy.squeeze()), data_info, batch_size, is_train=True)
    tf_train_it = TFIterWrapper(train_it, data_info, train_size, ceil(train_size/batch_size))

    x_test, y_test = shuffle_dataset(x_test, y_test.astype(np.int32))
    val_size = x_test.shape[0]
    val_it = _create_ds_and_iterator((x_test, y_test.squeeze()), data_info, batch_size, is_train=False)
    tf_val_it = TFIterWrapper(val_it, data_info, val_size, ceil(val_size/batch_size))

    verbose_data('train', x_train, y_train)
    verbose_data('test', x_test, y_test)

    return tf_train_it, tf_val_it


def get_cifar_lable_noise_datasets(dataset_name, batch_size, target_ratio, seed, noise_type='uniform', data_ratio=1.0):
    """Unified entry point for CIFAR-10 or CIFAR-100 noisy label datasets.

    Args:
      dataset_name: 'cifar10' or 'cifar100'.
      batch_size: Batch size.
      target_ratio: Noise ratio.
      seed: Random seed.
      noise_type: 'uniform' or 'asymmetric'.
      data_ratio: Fraction of training data to use.

    Returns:
      tf_train_it, tf_val_it: Training and validation iterator wrappers.
    """
    if dataset_name == 'cifar100':
        return get_cifar100_lable_noise_datasets(batch_size, target_ratio, seed, noise_type, data_ratio)
    elif dataset_name == 'cifar10':
        return get_cifar10_lable_noise_datasets(batch_size, target_ratio, seed, noise_type, data_ratio)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'cifar10' or 'cifar100'.")