# PySpark optimisation log

Every number below comes from `src/features/build_features.py` on the full M5
dataset (58.3M rows after the unpivot, 3049 SKUs x 10 stores x 1913 days) on a
single 8-core / 32 GB machine running `local[*]`, Spark 3.5.1.

Reproduce a timing with:

```bash
python -m src.features.build_features --config conf/config.yaml         # timed run
python -m src.features.build_features --config conf/config.yaml --explain  # physical plan
```

> Fill the table in from your own runs before you talk about it. The point of this
> document is that each row names one change and its measured effect - a list of
> tuning tips without numbers is worth nothing.

| # | Change | Wall clock | Shuffle read | Notes |
|---|--------|-----------:|-------------:|-------|
| 0 | Naive baseline: `unionAll` loop for the unpivot, default configs | _TBD_ | _TBD_ | Plan depth grows with the number of day columns |
| 1 | `stack()` instead of the union loop | _TBD_ | _TBD_ | One narrow transformation; the plan stops growing |
| 2 | Broadcast the calendar join | _TBD_ | _TBD_ | ~2k rows on the small side, no reason to shuffle the fact table |
| 3 | Parquet + partition by `store_id` | _TBD_ | _TBD_ | Predicate and partition pruning for every downstream read |
| 4 | Single `repartition` before the window functions | _TBD_ | _TBD_ | ~20 window functions then run without a shuffle between them |
| 5 | `spark.sql.shuffle.partitions` 200 -> tuned | _TBD_ | _TBD_ | Default over-provisions on a single node |
| 6 | AQE on (coalesce + skew join) | _TBD_ | _TBD_ | Lets Spark fix partition sizing at runtime |
| 7 | Salt the two hot store keys | _TBD_ | _TBD_ | See below |
| 8 | Arrow enabled for the pandas paths | _TBD_ | _TBD_ | Serialisation, not compute |

## The changes that mattered

### 1. `stack()` over a union loop

The M5 sales file is one row per series with 1913 `d_*` columns. The obvious
reshape is a loop of `unionAll` over the day columns. That builds a 1913-deep
logical plan; Catalyst spends longer optimising it than the job spends running, and
on a small driver it stack-overflows outright.

`stack(n, 'd_1', d_1, ...)` expresses the same reshape as a single expression, so
the plan stays flat regardless of how many days the dataset has.

See `src/ingestion/to_bronze.py:unpivot_sales`.

### 2. Broadcast the small side

`calendar` is ~1900 rows and `sell_prices` is ~6.8M. Joining calendar to the
58M-row fact table with a sort-merge join shuffles both sides. `F.broadcast()` ships
the small side to every executor and removes the shuffle entirely.

`spark.sql.autoBroadcastJoinThreshold` is set to 64MB in `spark_session.py` so AQE
also converts qualifying joins at runtime, but the explicit hint documents intent
and survives a bad statistics estimate.

### 3. Repartition once, then run every window function

Each lag and rolling feature is a window over `PARTITION BY store_id, item_id ORDER
BY date`. If the data is not already partitioned that way, every single window
triggers its own shuffle - roughly 20 shuffles of the full table.

`build_features.py` does one `repartition(n, "store_id", "item_id")` followed by
`sortWithinPartitions`, and all the windows then execute on already-correctly-placed
data. This is the largest single win in the pipeline.

### 4. Skew: two stores hold a disproportionate share of rows

`CA_1` and `CA_3` are materially larger than the other stores. Any per-store shuffle
lands most of the data on two tasks: the stage shows 8 tasks finishing in seconds
and 2 running for minutes, and the stage is only as fast as its slowest task.

Two mitigations, in order of preference:

1. `spark.sql.adaptive.skewJoin.enabled=true` - AQE splits oversized partitions
   automatically. This handles most cases and costs nothing.
2. Explicit salting (`transforms.salted_key`) for the cases AQE cannot see, e.g. a
   skewed `groupBy` rather than a join. Only the known-hot keys are salted, so the
   small side of the join is not inflated for the other 8 stores.

Salting is not free: the salted key has to be un-salted afterwards, and the small
side has to be exploded `salt_buckets` times. Reach for AQE first and salt only what
is still slow, with the Spark UI open to prove it.

### 5. What did not help

Notes on the things that looked promising and were not, so the next person does not
repeat them:

- _Caching the feature table before the window functions_: it is read once. Caching
  cost memory that the shuffle needed and made the stage slower.
- _Increasing `shuffle.partitions` further_: past the tuned value, task scheduling
  overhead dominated the parallelism gain on a single node.

## How to read the Spark UI for this job

1. **Stages tab** - find the stage with the widest gap between median and max task
   duration. That gap is skew, and it is where the time goes.
2. **SQL tab** - open the query plan and look for `Exchange` nodes. Each one is a
   shuffle; the goal of every change above is to remove one or make it smaller.
3. **Storage tab** - if anything is cached that is read once, un-cache it.
