"""
Module 1e — Delta Lake Writer
==============================

Helper class that wraps PySpark + delta-spark to provide a clean interface
for the three Delta tables used by NammaClimaGrid:

  Table               Schema
  ─────────────────── ──────────────────────────────────────────────
  raw_iot             ward_id, sensor_id, timestamp, lat, lon,
                      temperature_c, humidity_pct, pm25, rainfall_mm
  ward_features       ward_id, window_start, window_end, avg_lst,
                      avg_ndvi, avg_impervious_pct, avg_rainfall,
                      flood_report_count, avg_pm25
  satellite_features  ward_id, scene_date, ndvi, ndbi, impervious_pct,
                      lst_celsius, source_scene

Key features:
  * write_streaming() — append micro-batches (called from Flink/Structured Streaming)
  * write_batch()     — overwrite or merge a full batch (Spark jobs)
  * read_latest()     — returns a Spark DataFrame of the latest snapshot
  * time_travel()     — returns a specific Delta version for audit / replay

Usage:
    from delta_lake_writer import DeltaLakeWriter, get_spark
    spark = get_spark()
    writer = DeltaLakeWriter(spark, root="/data/delta")
    writer.write_batch("satellite_features", df)
    latest = writer.read_latest("satellite_features")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class DeltaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    delta_lake_root: str = "/data/delta"
    delta_table_raw_iot: str = "raw_iot"
    delta_table_ward_features: str = "ward_features"
    delta_table_satellite_features: str = "satellite_features"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "namma_clima_grid"
    postgres_user: str = "ncg_user"
    postgres_password: str = "change_me_in_production"


_settings: Optional[DeltaSettings] = None


def get_settings() -> DeltaSettings:
    global _settings
    if _settings is None:
        _settings = DeltaSettings()
    return _settings


# ---------------------------------------------------------------------------
# Spark session factory
# ---------------------------------------------------------------------------
def get_spark(app_name: str = "NammaClimaGrid", delta_root: Optional[str] = None):
    """
    Create (or reuse) a SparkSession with the Delta Lake extension configured.

    The Delta Lake extensions are loaded from the packages declared in the
    SparkSession builder so no manual JAR management is needed.
    """
    try:
        from delta import configure_spark_with_delta_pip  # type: ignore
        from pyspark.sql import SparkSession
    except ImportError as e:
        raise RuntimeError(
            "pyspark and delta-spark are required. "
            "Run: pip install pyspark==3.5.1 delta-spark==3.2.0"
        ) from e

    root = delta_root or get_settings().delta_lake_root
    Path(root).mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.warehouse.dir", root)
        # Keep local dev footprint small
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        # Avoid shuffle overhead for small datasets
        .config("spark.sql.shuffle.partitions", "4")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"SparkSession ready | version={spark.version} | warehouse={root}")
    return spark


# ---------------------------------------------------------------------------
# DeltaLakeWriter
# ---------------------------------------------------------------------------
class DeltaLakeWriter:
    """
    Manage writes and reads for the three NammaClimaGrid Delta tables.

    Args:
        spark:  active SparkSession (from get_spark())
        root:   local or HDFS/GCS path that acts as the Delta warehouse root.
    """

    TABLE_NAMES = ("raw_iot", "ward_features", "satellite_features")

    def __init__(self, spark, root: Optional[str] = None) -> None:
        self.spark = spark
        cfg = get_settings()
        self.root = Path(root or cfg.delta_lake_root)
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info(f"DeltaLakeWriter initialised | root={self.root}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _path(self, table: str) -> str:
        """Resolve absolute Delta table path."""
        if table not in self.TABLE_NAMES:
            raise ValueError(f"Unknown table '{table}'. Must be one of {self.TABLE_NAMES}")
        return str(self.root / table)

    def _delta(self, table: str):
        """Return a DeltaTable handle for merge / history operations."""
        from delta.tables import DeltaTable  # type: ignore
        return DeltaTable.forPath(self.spark, self._path(table))

    # ------------------------------------------------------------------
    # Write — streaming micro-batch
    # ------------------------------------------------------------------
    def write_streaming(
        self,
        df,
        table: str,
        checkpoint_dir: Optional[str] = None,
        trigger_interval: str = "60 seconds",
    ):
        """
        Write a Structured Streaming DataFrame to a Delta table.

        Designed to be called with a Spark Structured Streaming DataFrame that
        reads from Kafka. The stream runs in the background; call .awaitTermination()
        on the returned StreamingQuery to block.

        Args:
            df:               Streaming DataFrame.
            table:            Target table name.
            checkpoint_dir:   Checkpoint location (default: <root>/_checkpoints/<table>).
            trigger_interval: Trigger processing interval string.

        Returns:
            pyspark.sql.streaming.StreamingQuery
        """
        chk = checkpoint_dir or str(self.root / "_checkpoints" / table)
        Path(chk).mkdir(parents=True, exist_ok=True)

        query = (
            df.writeStream.format("delta")
            .outputMode("append")
            .option("checkpointLocation", chk)
            .trigger(processingTime=trigger_interval)
            .start(self._path(table))
        )
        logger.info(
            f"Streaming write started | table={table} | "
            f"checkpoint={chk} | trigger={trigger_interval}"
        )
        return query

    # ------------------------------------------------------------------
    # Write — batch
    # ------------------------------------------------------------------
    def write_batch(
        self,
        table: str,
        df,
        mode: str = "append",
        merge_key: Optional[str] = None,
    ) -> None:
        """
        Write a batch DataFrame to a Delta table.

        Args:
            table:      Target table name.
            df:         Spark DataFrame to write.
            mode:       'append' | 'overwrite'. Ignored when merge_key is set.
            merge_key:  If set, performs an upsert (MERGE) on this column
                        instead of a blind append/overwrite.
        """
        path = self._path(table)
        from delta.tables import DeltaTable  # type: ignore

        if merge_key and DeltaTable.isDeltaTable(self.spark, path):
            logger.info(f"MERGE into {table} on key={merge_key}")
            (
                self._delta(table)
                .alias("target")
                .merge(
                    df.alias("source"),
                    f"target.{merge_key} = source.{merge_key}",
                )
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
        else:
            logger.info(f"Writing {mode} to {table}")
            df.write.format("delta").mode(mode).save(path)

        count = self.spark.read.format("delta").load(path).count()
        logger.success(f"Write complete | table={table} | total_rows={count:,}")

    # ------------------------------------------------------------------
    # Read — latest snapshot
    # ------------------------------------------------------------------
    def read_latest(self, table: str):
        """
        Read the latest version of a Delta table as a Spark DataFrame.

        Returns:
            pyspark.sql.DataFrame
        """
        path = self._path(table)
        df = self.spark.read.format("delta").load(path)
        logger.info(
            f"Read latest | table={table} | rows={df.count():,} | "
            f"columns={df.columns}"
        )
        return df

    # ------------------------------------------------------------------
    # Read — time travel
    # ------------------------------------------------------------------
    def time_travel(self, table: str, version: int):
        """
        Read a specific historical version of a Delta table.

        Args:
            table:   Table name.
            version: Integer Delta version to read (0 = initial write).

        Returns:
            pyspark.sql.DataFrame at the requested version.
        """
        path = self._path(table)
        df = self.spark.read.format("delta").option("versionAsOf", version).load(path)
        logger.info(f"Time travel | table={table} | version={version} | rows={df.count():,}")
        return df

    # ------------------------------------------------------------------
    # Table history / metadata
    # ------------------------------------------------------------------
    def history(self, table: str, limit: int = 10):
        """Return the Delta transaction log history as a Pandas DataFrame."""
        hist_df = self._delta(table).history(limit)
        return hist_df.toPandas()

    def vacuum(self, table: str, retention_hours: int = 168) -> None:
        """
        Remove files older than `retention_hours` from the Delta table.
        Default retention is 7 days (168 hours).
        """
        logger.info(f"Vacuuming {table} | retention={retention_hours}h")
        self._delta(table).vacuum(retention_hours)


# ---------------------------------------------------------------------------
# Schema helpers — create empty Delta tables with enforced schema
# ---------------------------------------------------------------------------
def initialise_tables(writer: DeltaLakeWriter) -> None:
    """
    Create the three Delta tables with their canonical schemas if they don't
    already exist. Safe to call repeatedly (no-ops if tables exist).
    """
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    raw_iot_schema = StructType([
        StructField("ward_id",       IntegerType(), False),
        StructField("sensor_id",     StringType(),  False),
        StructField("timestamp",     StringType(),  False),
        StructField("lat",           DoubleType(),  True),
        StructField("lon",           DoubleType(),  True),
        StructField("temperature_c", DoubleType(),  True),
        StructField("humidity_pct",  DoubleType(),  True),
        StructField("pm25",          DoubleType(),  True),
        StructField("rainfall_mm",   DoubleType(),  True),
    ])

    ward_features_schema = StructType([
        StructField("ward_id",          IntegerType(), False),
        StructField("window_start",     StringType(),  False),
        StructField("window_end",       StringType(),  False),
        StructField("avg_temperature_c",DoubleType(),  True),
        StructField("avg_humidity_pct", DoubleType(),  True),
        StructField("avg_pm25",         DoubleType(),  True),
        StructField("sum_rainfall_mm",  DoubleType(),  True),
        StructField("reading_count",    IntegerType(), True),
        StructField("has_anomaly",      IntegerType(), True),
    ])

    satellite_features_schema = StructType([
        StructField("ward_id",        IntegerType(), False),
        StructField("scene_date",     StringType(),  False),
        StructField("ndvi",           DoubleType(),  True),
        StructField("ndbi",           DoubleType(),  True),
        StructField("impervious_pct", DoubleType(),  True),
        StructField("lst_celsius",    DoubleType(),  True),
        StructField("source_scene",   StringType(),  True),
    ])

    from delta.tables import DeltaTable  # type: ignore

    for table_name, schema in [
        ("raw_iot", raw_iot_schema),
        ("ward_features", ward_features_schema),
        ("satellite_features", satellite_features_schema),
    ]:
        path = writer._path(table_name)
        if not DeltaTable.isDeltaTable(writer.spark, path):
            empty_df = writer.spark.createDataFrame([], schema)
            empty_df.write.format("delta").save(path)
            logger.success(f"Created Delta table: {table_name} @ {path}")
        else:
            logger.debug(f"Table already exists: {table_name}")


# ---------------------------------------------------------------------------
# __main__ — quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from datetime import datetime, timezone

    from pyspark.sql import Row
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    logger.info("DeltaLakeWriter smoke test")
    spark = get_spark(app_name="DeltaTest", delta_root="/tmp/ncg_delta_test")
    writer = DeltaLakeWriter(spark, root="/tmp/ncg_delta_test")
    initialise_tables(writer)

    # Write a few mock raw_iot rows
    raw_iot_schema = StructType([
        StructField("ward_id",       IntegerType(), False),
        StructField("sensor_id",     StringType(),  False),
        StructField("timestamp",     StringType(),  False),
        StructField("lat",           DoubleType(),  True),
        StructField("lon",           DoubleType(),  True),
        StructField("temperature_c", DoubleType(),  True),
        StructField("humidity_pct",  DoubleType(),  True),
        StructField("pm25",          DoubleType(),  True),
        StructField("rainfall_mm",   DoubleType(),  True),
    ])
    rows = [
        Row(
            ward_id=12,
            sensor_id="SENSOR-0001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            lat=12.9756,
            lon=77.6050,
            temperature_c=33.4,
            humidity_pct=62.1,
            pm25=48.7,
            rainfall_mm=0.0,
        ),
        Row(
            ward_id=27,
            sensor_id="SENSOR-0002",
            timestamp=datetime.now(timezone.utc).isoformat(),
            lat=12.9352,
            lon=77.6245,
            temperature_c=31.8,
            humidity_pct=65.3,
            pm25=52.1,
            rainfall_mm=0.2,
        ),
    ]
    df = spark.createDataFrame(rows, schema=raw_iot_schema)
    writer.write_batch("raw_iot", df, mode="append")

    # Read back
    result = writer.read_latest("raw_iot")
    result.show()

    # Time travel to version 0
    v0 = writer.time_travel("raw_iot", version=0)
    logger.info(f"Version 0 row count: {v0.count()}")

    # History
    hist = writer.history("raw_iot", limit=3)
    logger.info(f"History:\n{hist[['version','timestamp','operation']].to_string()}")

    spark.stop()
    logger.success("DeltaLakeWriter smoke test passed.")
