import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromText, WriteToText
from typing import NamedTuple, List
import datetime
import logging

# Define data structure
class MLData(NamedTuple):
    id: str
    features: List[float]
    label: int
    timestamp: datetime.datetime

# Custom PTransform for data cleaning
class CleanMLData(beam.PTransform):
    def expand(self, pcoll):
        return (
            pcoll
            | 'FilterInvalid' >> beam.Filter(lambda x: len(x.features) > 0)
            | 'RemoveOutliers' >> beam.Filter(lambda x: all(-100 < f < 100 for f in x.features))
            | 'AddTimestamp' >> beam.Map(lambda x: x._replace(
                timestamp=datetime.datetime.now())
            )
        )

# Custom PTransform for feature engineering
class FeatureEngineering(beam.PTransform):
    def expand(self, pcoll):
        return (
            pcoll
            | 'NormalizeFeatures' >> beam.Map(
                lambda x: x._replace(
                    features=[f/max(x.features) for f in x.features]
                )
            )
            | 'AddDerivedFeatures' >> beam.Map(
                lambda x: x._replace(
                    features=x.features + [sum(x.features)/len(x.features)]
                )
            )
        )

# Parse input data
def parse_input(line: str) -> MLData:
    """Parse input line to MLData object"""
    try:
        items = line.split(',')
        return MLData(
            id=items[0],
            features=[float(f) for f in items[1:-1]],
            label=int(items[-1]),
            timestamp=None
        )
    except Exception as e:
        logging.error(f"Error parsing line: {line}, Error: {str(e)}")
        return None

# Format output data
def format_output(ml_data: MLData) -> str:
    """Format MLData object to output string"""
    features_str = ','.join(map(str, ml_data.features))
    return f"{ml_data.id},{features_str},{ml_data.label},{ml_data.timestamp}"

def run_pipeline():
    # Pipeline options
    options = PipelineOptions([
        '--runner=DirectRunner',
        '--project=your-project-id',
        '--region=your-region',
        '--temp_location=gs://your-bucket/temp',
    ])

    # Create pipeline
    with beam.Pipeline(options=options) as pipeline:
        # Extract: Read data from input source
        raw_data = (
            pipeline
            | 'ReadData' >> ReadFromText('input_data.csv', skip_header_lines=1)
            | 'ParseInput' >> beam.Map(parse_input)
            | 'FilterNone' >> beam.Filter(lambda x: x is not None)
        )

        # Transform: Clean and process data
        processed_data = (
            raw_data
            | 'CleanData' >> CleanMLData()
            | 'EngineerFeatures' >> FeatureEngineering()
        )

        # Load: Write results
        (
            processed_data
            | 'FormatOutput' >> beam.Map(format_output)
            | 'WriteResults' >> WriteToText(
                'output_data',
                file_name_suffix='.csv',
                header='id,features,label,timestamp'
            )
        )

        # Optional: Write statistics
        stats = (
            processed_data
            | 'CalculateStats' >> beam.CombineGlobally(
                lambda records: {
                    'count': len(list(records)),
                    'avg_features': sum(len(r.features) for r in records) / len(list(records))
                }
            )
            | 'WriteStats' >> WriteToText('stats.txt')
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run_pipeline()