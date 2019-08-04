import tensorflow.python.keras as k
import tensorflow as tf
import tensorflow.python.client.timeline as timeline
from model_extraction.tensorflow_utils import get_dummy_input_output


model = k.applications.VGG16(weights=None, include_top=True)
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
model.compile(loss="mean_squared_error", optimizer="adam", run_metadata=run_metadata, options=run_options)
x, y = get_dummy_input_output(model, 8)
model.fit(x, y, steps_per_epoch=2)
tl = timeline.Timeline(step_stats=run_metadata.step_stats)
with open("tensorflow_keras.chrometrace.json", "w") as file:
    file.write(tl.generate_chrome_trace_format())