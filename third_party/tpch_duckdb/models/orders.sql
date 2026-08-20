{{ config(materialized='table') }}
select
  o.o_orderkey      as order_id,
  o.o_orderkey      as order_key,
  o.o_orderdate     as order_date,
  o.o_orderstatus   as order_status,
  o.o_orderpriority as order_priority,
  o.o_totalprice    as total_price,
  c.c_mktsegment    as market_segment,
  c.c_name          as customer_name,
  n.n_name          as nation
from raw_orders o
join raw_customer c on o.o_custkey = c.c_custkey
join raw_nation n on c.c_nationkey = n.n_nationkey
