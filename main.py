import json
import os
import shutil
from argparse import ArgumentParser

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, window, col, udf, to_json, struct
from pyspark.sql.types import *


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-k', '--kafka-endpoint', type=str, default='localhost:9092', help='host:port for kafka')
    arg_parser.add_argument('--kafka-topic-traffic', type=str, default='traffic', help='topic to listen to for traffic')
    arg_parser.add_argument('--kafka-topic-tl', type=str, default='tl-status', help='topic to publish TL statuses to')

    arg_parser.add_argument('--spark-checkpoint-path', type=str, default='/tmp/spark/checkpoints', help='directory path for spark checkpoints')

    arg_parser.add_argument('--regular-from', type=int, default=4, help='lower boundary for regular vehicle count')
    arg_parser.add_argument('--regular-to', type=int, default=8, help='upper boundary for regular vehicle count')
    arg_parser.add_argument('--intersections', type=str, help='comma-separated list of SUMO intersection IDs to apply traffic control to')
    arg_parser.add_argument('--window-size', type=int, default=15, help='window size in seconds')
    arg_parser.add_argument('--window-slide', type=int, default=5, help='window slide in seconds')
    arg_parser.add_argument('--watermark', type=int, default=15, help='watermark in seconds')
    arg_parser.add_argument('--still-threshold', type=float, default=1.0, help='number of seconds (as decimal number) after which the car is treated as being still')
    args = arg_parser.parse_args()

    spark_checkpoint_path = args.spark_checkpoint_path

    if os.path.exists(spark_checkpoint_path):
        shutil.rmtree(spark_checkpoint_path)

    with open('sumo_map/junctions.json') as f:
        junctions = json.loads(f.read())

    with open('sumo_map/lanes.json') as f:
        lanes = json.loads(f.read())

    spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName('Traffic Stream Analysis') \
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

    lane_id_to_traffic_light = udf(lambda lane_id: {
        'id': lanes[lane_id]['tl_id'],
        'green_phases': lanes[lane_id]['green_phases']
    }, StructType({
        StructField('id', StringType()),
        StructField('green_phases', ArrayType(IntegerType()))
    }))

    count_to_status = udf(lambda count:
                          'none' if count == 0 else
                          'few' if 1 <= count <= args.regular_from - 1 else
                          'regular' if args.regular_from <= count <= args.regular_to else
                          'many')

    intersections = args.intersections.split(',') if args.intersections else None
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

    if intersections:
        filtered_traffic = traffic_next_junction \
            .where(col('next_junction').getField('id').isin(intersections))
    else:
        filtered_traffic = traffic_next_junction \
            .where(col('next_junction').getField('type') == 'traffic_light')

    counts_per_lane = filtered_traffic \
        .where(col('wait_time') > still_threshold) \
        .withColumn('timestamp', col('timestamp').cast(TimestampType())) \
        .withWatermark('timestamp', watermark) \
        .groupBy(window('timestamp', window_size, window_slide), col('next_junction').getField('id'), col('lane_id')) \
        .agg(F.approx_count_distinct('car_id').alias('count'), F.avg('wait_time').alias('avg_wait_time'))

    tl_statuses = counts_per_lane \
        .withColumn('traffic_light', lane_id_to_traffic_light(col('lane_id'))) \
        .withColumn('status', count_to_status(col('count')))

    tl_statuses \
        .select(struct('traffic_light', 'status').alias('value_as_struct')) \
        .select(to_json('value_as_struct').alias('value')) \
        .writeStream \
        .format('kafka') \
        .option('kafka.bootstrap.servers', args.kafka_endpoint) \
        .option('topic', args.kafka_topic_tl) \
        .option('checkpointLocation', spark_checkpoint_path) \
        .start() \
        .awaitTermination()

    spark.stop()


if __name__ == '__main__':
    main()
