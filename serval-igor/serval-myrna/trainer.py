import tensorflow as tf
from tensorflow import logging

class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task, train_dir, model, reader, model_exporter,
               log_device_placement=True, max_steps=None,
               export_model_steps=1000):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(
        allow_soft_placement=True,log_device_placement=log_device_placement)
    self.model = model
    self.reader = reader
    self.model_exporter = model_exporter
    self.max_steps = max_steps
    self.max_steps_reached = False
    self.export_model_steps = export_model_steps
    self.last_model_export_step = 0

#     if self.is_master and self.task.index > 0:
#       raise StandardError("%s: Only one replica of master expected",
#                           task_as_string(self.task))

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    if self.is_master and start_new_model:
      self.remove_training_directory(self.train_dir)

    target, device_fn = self.start_server_if_distributed()

    meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

    with tf.Graph().as_default() as graph:

      if meta_filename:
        saver = self.recover_model(meta_filename)

      with tf.device(device_fn):
        if not meta_filename:
          saver = self.build_model(self.model, self.reader)

        global_step = tf.get_collection("global_step")[0]
        loss = tf.get_collection("loss")[0]
        predictions = tf.get_collection("predictions")[0]
        labels = tf.get_collection("labels")[0]
        train_op = tf.get_collection("train_op")[0]
        init_op = tf.global_variables_initializer()

    sv = tf.train.Supervisor(
        graph,
        logdir=self.train_dir,
        init_op=init_op,
        is_chief=self.is_master,
        global_step=global_step,
        save_model_secs=15 * 60,
        save_summaries_secs=120,
        saver=saver)

    logging.info("%s: Starting managed session.", task_as_string(self.task))
    with sv.managed_session(target, config=self.config) as sess:
      try:
        logging.info("%s: Entering training loop.", task_as_string(self.task))
        while (not sv.should_stop()) and (not self.max_steps_reached):
          batch_start_time = time.time()
          _, global_step_val, loss_val, predictions_val, labels_val = sess.run(
              [train_op, global_step, loss, predictions, labels])
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = labels_val.shape[0] / seconds_per_batch

          if self.max_steps and self.max_steps <= global_step_val:
            self.max_steps_reached = True

          if self.is_master and global_step_val % 10 == 0 and self.train_dir:
            eval_start_time = time.time()
            hit_at_one = eval_util.calculate_hit_at_one(predictions_val, labels_val)
            perr = eval_util.calculate_precision_at_equal_recall_rate(predictions_val,
                                                                      labels_val)
            gap = eval_util.calculate_gap(predictions_val, labels_val)
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time

            logging.info("training step " + str(global_step_val) + " | Loss: " + ("%.2f" % loss_val) +
              " Examples/sec: " + ("%.2f" % examples_per_second) + " | Hit@1: " +
              ("%.2f" % hit_at_one) + " PERR: " + ("%.2f" % perr) +
              " GAP: " + ("%.2f" % gap))

            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_Hit@1", hit_at_one),
                global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_Perr", perr), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_GAP", gap), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("global_step/Examples/Second",
                                  examples_per_second), global_step_val)
            sv.summary_writer.flush()

            # Exporting the model every x steps
            time_to_export = ((self.last_model_export_step == 0) or
                (global_step_val - self.last_model_export_step
                 >= self.export_model_steps))

            if self.is_master and time_to_export:
              self.export_model(global_step_val, sv.saver, sv.save_path, sess)
              self.last_model_export_step = global_step_val
          else:
            logging.info("training step " + str(global_step_val) + " | Loss: " +
              ("%.2f" % loss_val) + " Examples/sec: " + ("%.2f" % examples_per_second))
      except tf.errors.OutOfRangeError:
        logging.info("%s: Done training -- epoch limit reached.",
                     task_as_string(self.task))

    logging.info("%s: Exited training loop.", task_as_string(self.task))
    sv.Stop()

  def export_model(self, global_step_val, saver, save_path, session):

    # If the model has already been exported at this step, return.
    if global_step_val == self.last_model_export_step:
      return

    last_checkpoint = saver.save(session, save_path, global_step_val)

    model_dir = "{0}/export/step_{1}".format(self.train_dir, global_step_val)
    logging.info("%s: Exporting the model at step %s to %s.",
                 task_as_string(self.task), global_step_val, model_dir)

    self.model_exporter.export_model(
        model_dir=model_dir,
        global_step_val=global_step_val,
        last_checkpoint=last_checkpoint)

  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""

    if self.cluster:
      logging.info("%s: Starting trainer within cluster %s.",
                   task_as_string(self.task), self.cluster.as_dict())
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info(
          "%s: Removing existing train directory.",
          task_as_string(self.task))
      gfile.DeleteRecursively(train_dir)
    except:
      logging.error(
          "%s: Failed to delete directory " + train_dir +
          " when starting a new model. Please delete it manually and" +
          " try again.", task_as_string(self.task))

  def get_meta_filename(self, start_new_model, train_dir):
    if start_new_model:
      logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                   task_as_string(self.task))
      return None

    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint:
      logging.info("%s: No checkpoint file found. Building a new model.",
                   task_as_string(self.task))
      return None

    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("%s: No meta graph file found. Building a new model.",
                     task_as_string(self.task))
      return None
    else:
      return meta_filename

  def recover_model(self, meta_filename):
    logging.info("%s: Restoring from meta graph file %s",
                 task_as_string(self.task), meta_filename)
    return tf.train.import_meta_graph(meta_filename)

  def build_model(self, model, reader):
    """Find the model and build the graph."""

    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

    build_graph(reader=reader,
                 model=model,
                 optimizer_class=optimizer_class,
                 clip_gradient_norm=FLAGS.clip_gradient_norm,
                 train_data_pattern=FLAGS.train_data_pattern,
                 label_loss_fn=label_loss_fn,
                 base_learning_rate=FLAGS.base_learning_rate,
                 learning_rate_decay=FLAGS.learning_rate_decay,
                 learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                 regularization_penalty=FLAGS.regularization_penalty,
                 num_readers=FLAGS.num_readers,
                 batch_size=FLAGS.batch_size,
                 num_epochs=FLAGS.num_epochs)

    return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25)