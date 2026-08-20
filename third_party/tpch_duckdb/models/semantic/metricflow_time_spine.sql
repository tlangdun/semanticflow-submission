{{ config(materialized='table') }}
with spine as (
  select * from generate_series(date '1990-01-01', date '2000-01-01', interval 1 day) as t(date_day)
)
select date_day from spine
