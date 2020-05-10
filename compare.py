from argparse import ArgumentParser
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, window, col, udf
from pyspark.sql.types import *


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-k', '--kafka-endpoint', type=str, default='localhost:9092', help='host:port for kafka')
    arg_parser.add_argument('--kafka-topic-traffic', type=str, default='traffic', help='topic to listen to for traffic')
    arg_parser.add_argument('--window-size', type=int, default=15, help='window size in seconds')
    arg_parser.add_argument('--window-slide', type=int, default=5, help='window slide in seconds')
    arg_parser.add_argument('--watermark', type=int, default=15, help='watermark in seconds')
    arg_parser.add_argument('--still-threshold', type=float, default=1.0, help='number of seconds (as decimal number) after which the car is treated as being still')
    args = arg_parser.parse_args()

    with open('sumo_map/junctions.json') as f:
        junctions = json.loads(f.read())

    with open('sumo_map/lanes.json') as f:
        lanes = json.loads(f.read())

    spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName('Track Main Intersection') \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    df = spark \
        .readStream \
        .format('kafka') \
        .option('kafka.bootstrap.servers', args.kafka_endpoint) \
        .option('subscribe', args.kafka_topic_traffic) \
        .load()

    traffic_schema = StructType([
        StructField('car_id', StringType()),
        StructField('road_id', StringType()),
        StructField('lane_id', StringType()),
        StructField('speed', DoubleType()),
        StructField('lon', DoubleType()),
        StructField('lat', DoubleType()),
        StructField('wait_time', DoubleType()),
        StructField('timestamp', DoubleType())
    ])

    lane_to_next_junction = udf(lambda lane_id: junctions[lanes[lane_id]['next_junction']], StructType({
        StructField('id', StringType()),
        StructField('type', StringType())
    }))

    window_size = f'{args.window_size} seconds'
    window_slide = f'{args.window_slide} seconds'
    watermark = f'{args.watermark} seconds'
    still_threshold = args.still_threshold

    traffic = df \
        .select(col('value').cast(StringType())) \
        .select(from_json('value', traffic_schema).alias('row')) \
        .select('row.*')

    traffic_next_junction = traffic \
        .withColumn('next_junction', lane_to_next_junction(col('lane_id')))

    stats = traffic_next_junction \
        .where(col('next_junction').getField('id') == 'main_intersection') \
        .where(col('wait_time') > still_threshold) \
        .withColumn('timestamp', col('timestamp').cast(TimestampType())) \
        .withWatermark('timestamp', watermark) \
        .groupBy(col('next_junction').getField('id').alias('junction_id'), window('timestamp', window_size, window_slide)) \
        .agg(F.approx_count_distinct('car_id').alias('count')) \
        .select('count')

    stats \
        .writeStream \
        .format('console') \
        .start() \
        .awaitTermination()

    spark.stop()


if __name__ == '__main__':
    main()
