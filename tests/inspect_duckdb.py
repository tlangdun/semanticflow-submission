import duckdb
import os

db_path = "third_party/jaffle_shop_duckdb/jaffle_shop.duckdb"
if not os.path.exists(db_path):
    print("Database file not found!")
    exit(0)

con = duckdb.connect(db_path)
print("Catalogs/Databases:")
print(con.sql("SELECT * FROM information_schema.schemata").fetchall())
print("\nTables:")
try:
    print(con.sql("SELECT * FROM information_schema.tables").fetchall())
except Exception as e:
    print(e)

# Check if we can query jaffle_shop.main.orders
print("\nQuery check 'jaffle_shop.main.orders':")
try:
    con.sql("SELECT count(*) FROM jaffle_shop.main.orders").show()
except Exception as e:
    print(e)
    
print("\nQuery check 'main.orders':")
try:
    con.sql("SELECT count(*) FROM main.orders").show()
except Exception as e:
    print(e)
