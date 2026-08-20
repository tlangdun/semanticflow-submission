{{ config(materialized='table') }}
select
  (l.l_orderkey * 100 + l.l_linenumber) as line_item_id,
  l.l_orderkey                          as order_id,
  l.l_shipdate                          as ship_date,
  l.l_quantity                          as quantity,
  l.l_extendedprice                     as extended_price,
  l.l_discount                          as discount,
  l.l_extendedprice * (1 - l.l_discount) as revenue,
  l.l_returnflag                        as return_flag,
  c.c_mktsegment                        as market_segment,
  n.n_name                              as nation
from raw_lineitem l
join raw_orders o on l.l_orderkey = o.o_orderkey
join raw_customer c on o.o_custkey = c.c_custkey
join raw_nation n on c.c_nationkey = n.n_nationkey
