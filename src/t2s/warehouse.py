"""Seeded generator for a synthetic retail warehouse stored in DuckDB."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from t2s.config import Settings

REGIONS = [
    ("North", "Northland"),
    ("South", "Northland"),
    ("East", "Eastmark"),
    ("West", "Eastmark"),
    ("Central", "Midvale"),
    ("Coastal", "Midvale"),
]
DEPARTMENTS = {
    "Electronics": ["Phones", "Laptops", "Audio"],
    "Home": ["Kitchen", "Furniture", "Garden"],
    "Apparel": ["Menswear", "Womenswear", "Footwear"],
    "Grocery": ["Snacks", "Beverages", "Pantry"],
}
PRICE_RANGE = {"Electronics": (40, 1500), "Home": (10, 700), "Apparel": (8, 180), "Grocery": (1, 30)}
SUPPLIER_COUNTRIES = ["Northland", "Eastmark", "Midvale", "Southport", "Westria"]
SEGMENTS = ["Consumer", "Small Business", "Corporate"]
CHANNELS = ["Online", "In-Store", "Mobile App"]
STORE_TYPES = ["Flagship", "Mall", "Outlet", "Express"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "Gift Card", "Wallet"]
RETURN_REASONS = ["Damaged", "Wrong Size", "Not as Described", "Changed Mind"]
STATUSES = ["Completed", "Cancelled", "Pending"]

DDL = """
CREATE TABLE regions (region_id INTEGER PRIMARY KEY, region_name VARCHAR, country VARCHAR);
CREATE TABLE stores (
    store_id INTEGER PRIMARY KEY, store_name VARCHAR, region_id INTEGER REFERENCES regions(region_id),
    store_type VARCHAR, opened_date DATE
);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY, first_name VARCHAR, last_name VARCHAR, segment VARCHAR,
    region_id INTEGER REFERENCES regions(region_id), signup_date DATE, loyalty_member BOOLEAN
);
CREATE TABLE suppliers (supplier_id INTEGER PRIMARY KEY, supplier_name VARCHAR, country VARCHAR);
CREATE TABLE categories (category_id INTEGER PRIMARY KEY, category_name VARCHAR, department VARCHAR);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY, product_name VARCHAR,
    category_id INTEGER REFERENCES categories(category_id),
    supplier_id INTEGER REFERENCES suppliers(supplier_id), list_price DECIMAL(10, 2), unit_cost DECIMAL(10, 2)
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id),
    store_id INTEGER REFERENCES stores(store_id), order_date DATE, channel VARCHAR, status VARCHAR
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id), quantity INTEGER, unit_price DECIMAL(10, 2),
    discount DECIMAL(4, 2)
);
CREATE TABLE payments (
    payment_id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(order_id), payment_method VARCHAR,
    amount DECIMAL(12, 2), paid_date DATE
);
CREATE TABLE returns (
    return_id INTEGER PRIMARY KEY, order_item_id INTEGER REFERENCES order_items(order_item_id),
    return_date DATE, reason VARCHAR, refund_amount DECIMAL(12, 2)
);
"""

FIRST = [
    "Ana",
    "Ben",
    "Carla",
    "Dev",
    "Elena",
    "Femi",
    "Gus",
    "Hana",
    "Ivo",
    "Jade",
    "Kai",
    "Lena",
    "Milo",
    "Nia",
]
LAST = ["Alder", "Brook", "Cole", "Dunn", "Ember", "Frost", "Gale", "Hale", "Irwin", "Jett", "Knox", "Lyle"]


def _dates(rng: np.random.Generator, start: str, end: str, n: int) -> pd.Series:
    lo, hi = pd.Timestamp(start).value // 86_400_000_000_000, pd.Timestamp(end).value // 86_400_000_000_000
    days = rng.integers(lo, hi + 1, size=n)
    return pd.Series(pd.to_datetime(days, unit="D")).dt.date


def generate(seed: int = 7, n_customers: int = 3000, n_orders: int = 25000) -> dict[str, pd.DataFrame]:
    """Build every warehouse table as a DataFrame."""
    rng = np.random.default_rng(seed)
    regions = pd.DataFrame(
        [(i + 1, name, country) for i, (name, country) in enumerate(REGIONS)],
        columns=["region_id", "region_name", "country"],
    )
    n_stores = 30
    stores = pd.DataFrame(
        {
            "store_id": np.arange(1, n_stores + 1),
            "store_name": [f"Store {i:02d}" for i in range(1, n_stores + 1)],
            "region_id": rng.integers(1, len(REGIONS) + 1, n_stores),
            "store_type": rng.choice(STORE_TYPES, n_stores, p=[0.15, 0.35, 0.2, 0.3]),
            "opened_date": _dates(rng, "2012-01-01", "2022-12-31", n_stores),
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_customers + 1),
            "first_name": rng.choice(FIRST, n_customers),
            "last_name": rng.choice(LAST, n_customers),
            "segment": rng.choice(SEGMENTS, n_customers, p=[0.7, 0.2, 0.1]),
            "region_id": rng.integers(1, len(REGIONS) + 1, n_customers),
            "signup_date": _dates(rng, "2020-01-01", "2025-06-30", n_customers),
            "loyalty_member": rng.random(n_customers) < 0.35,
        }
    )
    n_suppliers = 15
    suppliers = pd.DataFrame(
        {
            "supplier_id": np.arange(1, n_suppliers + 1),
            "supplier_name": [f"Supplier {chr(65 + i)}" for i in range(n_suppliers)],
            "country": rng.choice(SUPPLIER_COUNTRIES, n_suppliers),
        }
    )
    cat_rows = [(name, dept) for dept, names in DEPARTMENTS.items() for name in names]
    categories = pd.DataFrame(
        [(i + 1, n, d) for i, (n, d) in enumerate(cat_rows)],
        columns=["category_id", "category_name", "department"],
    )
    n_products = 200
    cat_ids = rng.integers(1, len(categories) + 1, n_products)
    depts = categories.set_index("category_id").loc[cat_ids, "department"].to_numpy()
    lo = np.array([PRICE_RANGE[d][0] for d in depts])
    hi = np.array([PRICE_RANGE[d][1] for d in depts])
    list_price = np.round(lo + (hi - lo) * rng.beta(1.5, 4, n_products), 2)
    products = pd.DataFrame(
        {
            "product_id": np.arange(1, n_products + 1),
            "product_name": [
                f"{categories.loc[c - 1, 'category_name']} Item {i:03d}" for i, c in enumerate(cat_ids, 1)
            ],
            "category_id": cat_ids,
            "supplier_id": rng.integers(1, n_suppliers + 1, n_products),
            "list_price": list_price,
            "unit_cost": np.round(list_price * rng.uniform(0.45, 0.8, n_products), 2),
        }
    )
    activity = rng.pareto(1.5, n_customers) + 0.2
    orders = pd.DataFrame(
        {
            "order_id": np.arange(1, n_orders + 1),
            "customer_id": rng.choice(customers["customer_id"], n_orders, p=activity / activity.sum()),
            "store_id": rng.integers(1, n_stores + 1, n_orders),
            "order_date": _dates(rng, "2023-01-01", "2025-12-31", n_orders),
            "channel": rng.choice(CHANNELS, n_orders, p=[0.45, 0.35, 0.2]),
            "status": rng.choice(STATUSES, n_orders, p=[0.9, 0.06, 0.04]),
        }
    )
    items_per_order = rng.integers(1, 5, n_orders)
    oi_order = np.repeat(orders["order_id"].to_numpy(), items_per_order)
    n_items = len(oi_order)
    popularity = rng.pareto(1.2, n_products) + 0.1
    oi_product = rng.choice(products["product_id"], n_items, p=popularity / popularity.sum())
    base_price = products.set_index("product_id").loc[oi_product, "list_price"].to_numpy()
    order_items = pd.DataFrame(
        {
            "order_item_id": np.arange(1, n_items + 1),
            "order_id": oi_order,
            "product_id": oi_product,
            "quantity": rng.integers(1, 6, n_items),
            "unit_price": np.round(base_price * rng.uniform(0.95, 1.05, n_items), 2),
            "discount": rng.choice([0.0, 0.05, 0.1, 0.2], n_items, p=[0.6, 0.2, 0.15, 0.05]),
        }
    )
    line = order_items["quantity"] * order_items["unit_price"] * (1 - order_items["discount"])
    totals = line.groupby(order_items["order_id"]).sum().round(2)
    paid = orders[orders["status"] != "Cancelled"].set_index("order_id")
    pay_order = paid.index.to_numpy()
    payments = pd.DataFrame(
        {
            "order_id": pay_order,
            "payment_method": rng.choice(PAYMENT_METHODS, len(pay_order), p=[0.5, 0.3, 0.08, 0.12]),
            "amount": totals.loc[pay_order].to_numpy(),
            "paid_date": [
                d + pd.Timedelta(days=int(k))
                for d, k in zip(paid["order_date"], rng.integers(0, 3, len(paid)), strict=True)
            ],
        }
    )
    payments.insert(0, "payment_id", np.arange(1, len(payments) + 1))
    completed_ids = orders.loc[orders["status"] == "Completed", "order_id"]
    eligible = order_items[order_items["order_id"].isin(completed_ids)]
    ret = eligible.sample(frac=0.06, random_state=seed).sort_values("order_item_id")
    order_dates = orders.set_index("order_id").loc[ret["order_id"], "order_date"].to_numpy()
    returns = pd.DataFrame(
        {
            "return_id": np.arange(1, len(ret) + 1),
            "order_item_id": ret["order_item_id"].to_numpy(),
            "return_date": [
                d + pd.Timedelta(days=int(k))
                for d, k in zip(order_dates, rng.integers(3, 40, len(ret)), strict=True)
            ],
            "reason": rng.choice(RETURN_REASONS, len(ret), p=[0.25, 0.3, 0.2, 0.25]),
            "refund_amount": np.round(
                (ret["quantity"] * ret["unit_price"] * (1 - ret["discount"])).to_numpy(dtype=float), 2
            ),
        }
    )
    return {
        "regions": regions,
        "stores": stores,
        "customers": customers,
        "suppliers": suppliers,
        "categories": categories,
        "products": products,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "returns": returns,
    }


def build(db_path: Path, seed: int = 7) -> Path:
    """Write a fresh warehouse to ``db_path``."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    tables = generate(seed)
    con = duckdb.connect(str(db_path))
    con.execute(DDL)
    for name, df in tables.items():
        con.register("df_view", df)
        con.execute(f"INSERT INTO {name} SELECT * FROM df_view")
        con.unregister("df_view")
    con.close()
    return db_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Settings().db_path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    path = build(args.db, args.seed)
    con = duckdb.connect(str(path), read_only=True)
    for (name,) in con.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall():
        print(f"{name:12s} {con.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]:>8,d}")


if __name__ == "__main__":
    main()
