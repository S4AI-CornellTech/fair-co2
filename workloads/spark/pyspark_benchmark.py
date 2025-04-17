# Copyright 2018 Databricks, Inc.
#
# This work (the "Licensed Material") is licensed under the Creative Commons
# Attribution-NonCommercial-NoDerivatives 4.0 International License. You may
# not use this file except in compliance with the License.
#
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/
#
# Unless required by applicable law or agreed to in writing, the Licensed Material is offered
# on an "AS-IS" and "AS-AVAILABLE" BASIS, WITHOUT REPRESENTATIONS OR WARRANTIES OF ANY KIND,
# whether express, implied, statutory, or other. This includes, without limitation, warranties
# of title, merchantability, fitness for a particular purpose, non-infringement, absence of
# latent or other defects, accuracy, or the presence or absence of errors, whether or not known
# or discoverable. To the extent legally permissible, in no event will the Licensor be liable
# to You on any legal theory (including, without limitation, negligence) or otherwise for
# any direct, special, indirect, incidental, consequential, punitive, exemplary, or other
# losses, costs, expenses, or damages arising out of this License or use of the Licensed
# Material, even if the Licensor has been advised of the possibility of such losses, costs,
# expenses, or damages.
from pyspark.sql import SparkSession
from contextlib import contextmanager
import time
import logging
import argparse
import os

@contextmanager
def time_usage(name, itr, f):
    """log the time usage in a code block
    prefix: the prefix text to show
    """
    start = time.time()
    f.write(f'{name} round {itr} starting at {start}\n')
    f.flush()
    yield
    end = time.time()
    f.write(f'{name} round {itr} ended at {end}\n')
    elapsed_seconds = float("%.4f" % (end - start))
    f.write(f'{name} round {itr}: elapsed seconds: {elapsed_seconds}\n')
    f.flush()


logging.getLogger().setLevel(logging.INFO)

spark = SparkSession.builder.appName("pyspark_benchmark").getOrCreate()
parser = argparse.ArgumentParser(description='PySpark Query Benchmark')
parser.add_argument('--data_path',
                    help='Path of TPC-DS CSV file directory, e.g. /data/tpcds250')
parser.add_argument("--start_time")
parser.add_argument("--rounds")

args = parser.parse_args()
start_time = int(args.start_time)
rounds = int(args.rounds)

f = open('spark.txt', 'w')

# Read all the files...
store_sales_headers = ['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk', 'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk', 'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price', 'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax', 'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit', 'dummy']
store_sales_df = spark.read.csv(args.data_path, header=False).toDF(*store_sales_headers)

store_sales_df.createOrReplaceTempView("t")

# Warm up
spark.sql("select sum(ss_customer_sk) from t").collect()

curr_time = time.time()
sleep_time = start_time - curr_time
if sleep_time > 0:
    time.sleep(sleep_time)

# Loop to warm up JVM
for i in range(1, rounds + 1):
    with time_usage('sum', i, f):
        spark.sql("select sum(ss_customer_sk) from t").collect()
    # with time_usage('count distinct', i, f):
    #     spark.sql('select count(distinct ss_quantity) from t').collect()
    # with time_usage('aggregate', i, f):
    #     spark.sql("select sum(ss_net_profit) from t group by ss_store_sk").collect()
    # with time_usage('sum2', i, f):
    #     spark.sql("select sum(ss_sold_time_sk) from t").collect()
    # with time_usage('count distinct2', i, f):
    #     spark.sql('select count(distinct ss_hdemo_sk) from t').collect()
    # with time_usage('aggregate2', i, f):
    #     spark.sql("select sum(ss_list_price) from t group by ss_promo_sk").collect()
    # with time_usage('sum3', i, f):
    #     spark.sql("select sum(ss_item_sk) from t").collect()
    # with time_usage('aggregate3', i, f):
    #     spark.sql("select sum(ss_sales_price) from t group by ss_addr_sk").collect()
