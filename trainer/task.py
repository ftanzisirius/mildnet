from __future__ import absolute_import
from __future__ import print_function

needs_reproducible = True
if needs_reproducible:
    from numpy.random import seed

    seed(1)
    from tensorflow import random

    random.set_seed(2)

from .checkpointers import *
from .accuracy import *
from .utils import *
from .model import *
import inspect
import argparse
import pandas as pd
import dill
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import logging


def main(job_dir, data_path, model_id, weights_path, loss, train_csv, val_csv, batch_size, train_epocs, optimizer,
         is_tpu, lr, **args):
    logging.getLogger().setLevel(logging.INFO)

    if not os.path.exists("output"):
        os.makedirs("output")

    batch_size *= 3
    is_full_data = False

    logging.info("Downloading Training Image from path {}".format(data_path))
    downloads_training_images(data_path, is_cropped=("_cropped" in job_dir))

    logging.info("Building Model: {}".format(model_id))
    if model_id in globals():
        model_getter = globals()[model_id]
        model = model_getter()
    else:
        raise RuntimeError("Failed. Model function {} not found".format(model_id))

    if loss + "_fn" in globals():
        _loss_tensor = globals()[loss + "_fn"](batch_size)
    else:
        raise RuntimeError("Failed. Loss function {} not found".format(loss + "_fn"))

    accuracy = accuracy_fn(batch_size)
    img_width, img_height = [int(v) for v in model.input[0].shape[1:3]]

    dg = DataGenerator({
        "rescale": 1. / 255,
        "horizontal_flip": True,
        "vertical_flip": True,
        "zoom_range": 0.2,
        "shear_range": 0.2,
        "rotation_range": 30,
        "fill_mode": 'nearest'
    }, data_path, train_csv, val_csv, target_size=(img_width, img_height))

    train_generator = dg.get_train_generator(batch_size, is_full_data)
    test_generator = dg.get_test_generator(batch_size)

    if weights_path:
        with file_io.FileIO(weights_path, mode='r') as input_f:
            with file_io.FileIO("weights.h5", mode='w+') as output_f:
                output_f.write(input_f.read())
        model.load_weights("weights.h5")

    # model = multi_gpu_model(model, gpus=4)
    if optimizer == "mo":
        tf.compat.v1.disable_eager_execution()
        model.compile(loss=_loss_tensor,
                      optimizer=tf.compat.v1.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True),
                      metrics=[accuracy]
                      )
        model.run_eagerly = True
    elif optimizer == "rms":
        tf.compat.v1.disable_eager_execution()
        model.compile(loss=_loss_tensor,
                      optimizer=tf.compat.v1.train.RMSPropOptimizer(lr),
                      metrics=[accuracy]
                      )
        model.run_eagerly = True
    else:
        logging.error("Optimizer not supported")
        return

    csv_logger = CSVLogger(job_dir, "output/training.log")
    model_checkpoint_path = "weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
    model_checkpointer = ModelCheckpoint(job_dir, model_checkpoint_path, save_best_only=True, save_weights_only=True,
                                         monitor="val_loss", verbose=1)
    tensorboard = TensorBoard(log_dir=job_dir + '/logs/', histogram_freq=0, write_graph=True, write_images=True)
    # test_accuracy = TestAccuracy(data_path)  # Not using test data as of now

    callbacks = [csv_logger, model_checkpointer, tensorboard]

    model_json = model.to_json()
    write_file_and_backup(model_json, job_dir, "output/model.def")

    with open("output/model_code.pkl", 'wb') as f:
        dill.dump(model_getter, f)
    backup_file(job_dir, "output/model_code.pkl")

    model_code = inspect.getsource(model_getter)
    write_file_and_backup(model_code, job_dir, "output/model_code.txt")

    if is_tpu:
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'])
            )
        )

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=(train_generator.n // (train_generator.batch_size)),
                                  validation_data=test_generator,
                                  epochs=train_epocs,
                                  validation_steps=(test_generator.n // (test_generator.batch_size)),
                                  callbacks=callbacks)

    pd.DataFrame(history.history).to_csv("output/history.csv")
    backup_file(job_dir, "output/history.csv")

    model.save_weights('output/model.h5')
    backup_file(job_dir, 'output/model.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--data-path',
        help='GCS or local paths to training data, should contain images folder and triplets csv',
        required=True
    )
    parser.add_argument(
        '--optimizer',
        help='Optimizer',
        required=True
    )
    parser.add_argument(
        '--model-id',
        help='model id',
        required=True
    )
    parser.add_argument(
        '--weights-path',
        help='GCS location of pretrained weights path',
        default=None
    )
    parser.add_argument(
        '--loss',
        help='loss function',
        required=True
    )
    parser.add_argument(
        '--train-csv',
        help='train csv file name',
        default='tops_train_shuffle.csv'
    )
    parser.add_argument(
        '--val-csv',
        help='val csv file name',
        default='tops_val_full.csv'
    )
    parser.add_argument(
        '--batch-size',
        help='batch size',
        default=16,
        type=int
    )
    parser.add_argument(
        '--is-tpu',
        help='is tpu used',
        default=False,
        type=bool
    )
    parser.add_argument(
        '--train-epocs',
        help='number of epochs to train',
        default=6,
        type=int
    )
    parser.add_argument(
        '--lr',
        help='learning rate',
        default=0.001,
        type=float
    )

    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
