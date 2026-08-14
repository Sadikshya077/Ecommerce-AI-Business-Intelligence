-- warehouse/olap_queries.sql
-- Example OLAP operations against the star schema.
-- Run these in psql or any SQL client connected to ecommerce_bi.
-- Save results/screenshots of these for the OLAP section of your report.

-- ============================================================
-- ROLL-UP: total revenue by year -> quarter
-- ============================================================
SELECT d.year, d.quarter, ROUND(SUM(f.price)::numeric, 2) AS total_revenue
FROM fact_order_item f
JOIN dim_date d ON f.order_date_key = d.date_key
GROUP BY ROLLUP (d.year, d.quarter)
ORDER BY d.year, d.quarter;

-- ============================================================
-- DRILL-DOWN: revenue by state, then by city within one state
-- ============================================================
SELECT c.customer_state, c.customer_city, ROUND(SUM(f.price)::numeric, 2) AS revenue
FROM fact_order_item f
JOIN dim_customer c ON f.customer_key = c.customer_key
WHERE c.customer_state = 'SP'
GROUP BY c.customer_state, c.customer_city
ORDER BY revenue DESC;

-- ============================================================
-- SLICE: category revenue for Q4 only
-- ============================================================
SELECT p.category_name_english, ROUND(SUM(f.price)::numeric, 2) AS revenue
FROM fact_order_item f
JOIN dim_product p ON f.product_key = p.product_key
JOIN dim_date d ON f.order_date_key = d.date_key
WHERE d.quarter = 4
GROUP BY p.category_name_english
ORDER BY revenue DESC;

-- ============================================================
-- DICE: revenue for two states x two quarters
-- ============================================================
SELECT c.customer_state, d.quarter, ROUND(SUM(f.price)::numeric, 2) AS revenue
FROM fact_order_item f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_date d ON f.order_date_key = d.date_key
WHERE c.customer_state IN ('SP', 'RJ') AND d.quarter IN (1, 2)
GROUP BY c.customer_state, d.quarter
ORDER BY c.customer_state, d.quarter;

-- ============================================================
-- PIVOT-style: category x month revenue matrix
-- ============================================================
SELECT
    p.category_name_english,
    SUM(CASE WHEN d.month = 1 THEN f.price ELSE 0 END) AS jan,
    SUM(CASE WHEN d.month = 2 THEN f.price ELSE 0 END) AS feb,
    SUM(CASE WHEN d.month = 3 THEN f.price ELSE 0 END) AS mar,
    SUM(CASE WHEN d.month = 4 THEN f.price ELSE 0 END) AS apr
FROM fact_order_item f
JOIN dim_product p ON f.product_key = p.product_key
JOIN dim_date d ON f.order_date_key = d.date_key
WHERE d.month BETWEEN 1 AND 4
GROUP BY p.category_name_english
ORDER BY p.category_name_english;

-- ============================================================
-- Bonus: average delivery delay by seller state
-- (useful preview for churn feature engineering in Phase 3-4)
-- ============================================================
SELECT s.seller_state, ROUND(AVG(f.delivery_delay_days)::numeric, 2) AS avg_delay_days, COUNT(*) AS n_orders
FROM fact_order_item f
JOIN dim_seller s ON f.seller_key = s.seller_key
WHERE f.delivery_delay_days IS NOT NULL
GROUP BY s.seller_state
ORDER BY avg_delay_days DESC;
