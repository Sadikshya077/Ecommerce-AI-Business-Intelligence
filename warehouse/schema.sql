-- warehouse/schema.sql
-- Star schema for the Olist e-commerce data warehouse.
-- Fact_OrderItem grain: one row per (order, product, seller) line item.
--
-- Apply with:
--   psql -U postgres -d ecommerce_bi -f warehouse/schema.sql

DROP TABLE IF EXISTS fact_order_item CASCADE;
DROP TABLE IF EXISTS dim_customer CASCADE;
DROP TABLE IF EXISTS dim_product CASCADE;
DROP TABLE IF EXISTS dim_seller CASCADE;
DROP TABLE IF EXISTS dim_payment CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;
DROP TABLE IF EXISTS dim_review CASCADE;

CREATE TABLE dim_customer (
    customer_key         SERIAL PRIMARY KEY,
    customer_id           VARCHAR(64) UNIQUE NOT NULL,
    customer_unique_id     VARCHAR(64),
    customer_city           VARCHAR(128),
    customer_state           VARCHAR(8),
    customer_zip_prefix       VARCHAR(16)
);

CREATE TABLE dim_product (
    product_key           SERIAL PRIMARY KEY,
    product_id             VARCHAR(64) UNIQUE NOT NULL,
    category_name           VARCHAR(128),
    category_name_english     VARCHAR(128),
    weight_g                   NUMERIC,
    length_cm                   NUMERIC,
    height_cm                    NUMERIC,
    width_cm                      NUMERIC
);

CREATE TABLE dim_seller (
    seller_key             SERIAL PRIMARY KEY,
    seller_id               VARCHAR(64) UNIQUE NOT NULL,
    seller_city               VARCHAR(128),
    seller_state                VARCHAR(8),
    seller_zip_prefix             VARCHAR(16)
);

CREATE TABLE dim_payment (
    payment_key             SERIAL PRIMARY KEY,
    payment_type              VARCHAR(32) UNIQUE NOT NULL
);

CREATE TABLE dim_date (
    date_key                 INTEGER PRIMARY KEY,        -- YYYYMMDD
    full_date                  DATE NOT NULL,
    day                          SMALLINT,
    month                         SMALLINT,
    quarter                        SMALLINT,
    year                             SMALLINT,
    day_of_week                       SMALLINT,
    is_weekend                          BOOLEAN
);

CREATE TABLE dim_review (
    review_key                SERIAL PRIMARY KEY,
    order_id                    VARCHAR(64) UNIQUE NOT NULL,
    review_score                  SMALLINT
);

CREATE TABLE fact_order_item (
    fact_key                   BIGSERIAL PRIMARY KEY,
    order_id                     VARCHAR(64) NOT NULL,
    order_item_id                  SMALLINT NOT NULL,
    customer_key                     INTEGER REFERENCES dim_customer(customer_key),
    product_key                       INTEGER REFERENCES dim_product(product_key),
    seller_key                         INTEGER REFERENCES dim_seller(seller_key),
    payment_key                         INTEGER REFERENCES dim_payment(payment_key),
    order_date_key                       INTEGER REFERENCES dim_date(date_key),
    review_key                            INTEGER REFERENCES dim_review(review_key),

    price                                  NUMERIC(12, 2) NOT NULL,
    freight_value                            NUMERIC(12, 2),
    payment_value                              NUMERIC(12, 2),
    payment_installments                         SMALLINT,
    delivery_days                                  INTEGER,
    delivery_delay_days                              INTEGER,

    UNIQUE (order_id, order_item_id)
);

CREATE INDEX idx_fact_customer ON fact_order_item(customer_key);
CREATE INDEX idx_fact_product  ON fact_order_item(product_key);
CREATE INDEX idx_fact_seller   ON fact_order_item(seller_key);
CREATE INDEX idx_fact_date     ON fact_order_item(order_date_key);
CREATE INDEX idx_fact_payment  ON fact_order_item(payment_key);
CREATE INDEX idx_fact_review   ON fact_order_item(review_key);
