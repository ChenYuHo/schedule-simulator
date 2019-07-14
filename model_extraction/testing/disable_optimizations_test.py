import tensorflow.python.keras as k
import tensorflow as tf
import tensorflow.python.client.timeline as timeline
print("Tensorflow version: {}".format(tf.__version__))

if tf.__version__ == "1.13.1" or tf.__version__ == "1.14":
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config_proto = tf.ConfigProto(graph_options=graph_options)
    for name, sess in [("not-optimized", tf.Session(config=config_proto, graph=tf.Graph())),
                       ("optimized", tf.Session(graph=tf.Graph()))]:
        with sess:
            with tf.name_scope(name):
                a = tf.add(10, 20)
                b = tf.add(a, 10)
                c = tf.print(b)
                print(sess.graph.get_operations())
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run(c, options=run_options, run_metadata=run_metadata)
            with open("{}.trace".format(name), "w") as file:
                file.write(timeline.Timeline(step_stats=run_metadata.step_stats)
                           .generate_chrome_trace_format())

elif tf.__version__ == "2.0.0-beta1":
    optimizer_options = tf.compat.v1.OptimizerOptions(opt_level=tf.compat.v1.OptimizerOptions.L0)
    graph_options = tf.compat.v1.GraphOptions(optimizer_options=optimizer_options)
    config_proto = tf.compat.v1.ConfigProto(graph_options=graph_options)
    for name, sess in [("not-optimized", tf.compat.v1.Session(config=config_proto, graph=tf.Graph())),
                       ("optimized", tf.compat.v1.Session(graph=tf.Graph()))]:
        with sess:
            with tf.name_scope(name):
                a = tf.add(10, 20)
                b = tf.add(a, 10)
                c = tf.print(b)
                print(sess.graph.get_operations())
            run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
            sess.run(c, options=run_options, run_metadata=run_metadata)
            with open("{}.trace".format(name), "w") as file:
                file.write(timeline.Timeline(step_stats=run_metadata.step_stats)
                           .generate_chrome_trace_format())