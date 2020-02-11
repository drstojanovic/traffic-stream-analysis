import json
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, window, col, udf, to_json, struct
from pyspark.sql.types import *

CHECKPOINT_LOCATION = '/tmp/spark/checkpoints'


def main():
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
        .option('subscribe', 'nis-waittime') \
        .load()

    schema = StructType([
        StructField('car_id', StringType()),
        StructField('road_id', StringType()),
        StructField('lane_id', StringType()),
        StructField('speed', DoubleType()),
        StructField('lon', DoubleType()),
        StructField('lat', DoubleType()),
        StructField('wait_time', DoubleType()),
        StructField('timestamp', DoubleType())
    ])

    map_junction_to_lat_lon = udf(lambda junction_id: {
        'lat': junctions[junction_id]['lat'],
        'lon': junctions[junction_id]['lon']
    }, MapType(StringType(), DoubleType()))

    map_lane_id_to_next_junction = udf(lambda lane_id: lanes[lane_id]['next_junction'], StringType())

    map_lane_id_to_max_speed = udf(lambda lane_id: lanes[lane_id]['max_speed'], DoubleType())

    df \
        .selectExpr('CAST(value AS STRING)') \
        .select(from_json('value', schema).alias('row')) \
        .select('row.*') \
        .withColumn('speed_pct', col('speed') / map_lane_id_to_max_speed(col('lane_id'))) \
        .withColumn('next_junction', map_lane_id_to_next_junction(col('lane_id'))) \
        .withColumn('timestamp', col('timestamp').cast(TimestampType())) \
        .groupBy(window('timestamp', '30 seconds', '5 seconds'), col('next_junction')) \
        .agg(F.avg('speed_pct')).withColumnRenamed('avg(speed_pct)', 'avg_speed_pct') \
        .withColumn('next_junction', map_junction_to_lat_lon(col('next_junction'))) \
        .select(struct('window', 'next_junction', 'avg_speed_pct').alias('value_as_struct')) \
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
