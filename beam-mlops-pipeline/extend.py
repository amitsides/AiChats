# Add model training preparation
class PrepareForTraining(beam.PTransform):
    def expand(self, pcoll):
        return (
            pcoll
            | 'SplitFeaturesLabel' >> beam.Map(
                lambda x: {
                    'features': x.features,
                    'label': x.label
                }
            )
            | 'ConvertToTFExample' >> beam.Map(self.to_tf_example)
        )

    @staticmethod
    def to_tf_example(instance):
        import tensorflow as tf
        feature = {
            'features': tf.train.Feature(
                float_list=tf.train.FloatList(value=instance['features'])
            ),
            'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[instance['label']])
            )
        }
        return tf.train.Example(
            features=tf.train.Features(feature=feature)
        ).SerializeToString()

# Add to pipeline:
training_data = (
    processed_data
    | 'PrepareForTraining' >> PrepareForTraining()
    | 'WriteTFRecords' >> beam.io.WriteToTFRecord(
        'training_data',
        file_name_suffix='.tfrecord'
    )
)