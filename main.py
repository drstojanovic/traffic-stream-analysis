import json
import os
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, window, col, udf, to_json, struct
from pyspark.sql.types import *

CHECKPOINT_LOCATION = '/tmp/spark/checkpoints'
JUNCTIONS = None
LANES = None


def main():
    if os.path.exists(CHECKPOINT_LOCATION):
        shutil.rmtree(CHECKPOINT_LOCATION)

    with open('sumo_map/junctions.json') as f:
        junctions = json.loads(f.read())

    with open('sumo_map/lanes.json') as f:
        lanes = json.loads(f.read())

    spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName('Test') \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    df = spark \
        .readStream \
        .format('kafka') \
        .option('kafka.bootstrap.servers', 'localhost:9092') \
        .option('subscribe', 'traffic') \
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
    lane_id_to_max_speed = udf(lambda lane_id: lanes[lane_id]['max_speed'], DoubleType())
    junction_id_to_coords = udf(lambda junction_id: {
        'lat': junctions[junction_id]['lat'],
        'lon': junctions[junction_id]['lon']
    }, MapType(StringType(), DoubleType()))

    traffic = df \
        .select(col('value').cast(StringType())) \
        .select(from_json('value', traffic_schema).alias('row')) \
        .select('row.*')

    traffic \
        .withColumn('next_junction', lane_to_next_junction(col('lane_id'))) \
        .where(col('next_junction').getField('id') == 'cluster_5287628218_5287628220_5287628726_5287628727_6522223862_cluster_453204528_453204531') \
        .withColumn('speed_pct', col('speed') / lane_id_to_max_speed(col('lane_id'))) \
        .withColumn('timestamp', col('timestamp').cast(TimestampType())) \
        .withWatermark('timestamp', '30 seconds') \
        .groupBy(window('timestamp', '30 seconds', '5 seconds'), col('next_junction').getField('id').alias('junction_id')) \
        .agg(F.avg('speed_pct')).withColumnRenamed('avg(speed_pct)', 'avg_speed_pct') \
        .withColumn('coords', junction_id_to_coords(col('junction_id'))) \
        .select(struct('junction_id', 'coords', 'avg_speed_pct').alias('value_as_struct')) \
        .select(to_json('value_as_struct').alias('value')) \
        .writeStream \
        .format('kafka') \
        .option('kafka.bootstrap.servers', 'localhost:9092') \
        .option('topic', 'results') \
        .option('checkpointLocation', CHECKPOINT_LOCATION) \
        .start() \
        .awaitTermination()

        # .writeStream \
        # .option('truncate', 0) \
        # .option('numRows', 10) \
        # .outputMode('complete') \
        # .format('console') \
        # .start() \
        # .awaitTermination()


if __name__ == '__main__':
    main()
