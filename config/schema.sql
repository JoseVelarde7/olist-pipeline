-- Schema para base de datos Olist en MariaDB
-- Usar la base de datos existente
USE olist;

-- Tabla de geolocalización
CREATE TABLE olist_geolocation (
    id INT AUTO_INCREMENT PRIMARY KEY,
    geolocation_zip_code_prefix VARCHAR(10) NOT NULL,
    geolocation_lat DECIMAL(10, 8),
    geolocation_lng DECIMAL(11, 8),
    geolocation_city VARCHAR(100),
    geolocation_state VARCHAR(2),
    INDEX idx_zip_code (geolocation_zip_code_prefix)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de clientes
CREATE TABLE olist_customers (
    customer_id VARCHAR(64) PRIMARY KEY,
    customer_unique_id VARCHAR(64),
    customer_zip_code_prefix VARCHAR(10),
    customer_city VARCHAR(100),
    customer_state VARCHAR(2),
    INDEX idx_unique_id (customer_unique_id),
    INDEX idx_zip_code (customer_zip_code_prefix)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de vendedores
CREATE TABLE olist_sellers (
    seller_id VARCHAR(64) PRIMARY KEY,
    seller_zip_code_prefix VARCHAR(10),
    seller_city VARCHAR(100),
    seller_state VARCHAR(2),
    INDEX idx_zip_code (seller_zip_code_prefix)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de traducción de categorías de productos
CREATE TABLE product_category_translation (
    product_category_name VARCHAR(100) PRIMARY KEY,
    product_category_name_english VARCHAR(100)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de productos
CREATE TABLE olist_products (
    product_id VARCHAR(64) PRIMARY KEY,
    product_category_name VARCHAR(100),
    product_name_lenght INT,
    product_description_lenght INT,
    product_photos_qty INT,
    product_weight_g INT,
    product_length_cm INT,
    product_height_cm INT,
    product_width_cm INT,
    INDEX idx_category (product_category_name),
    FOREIGN KEY (product_category_name) REFERENCES product_category_translation(product_category_name)
        ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de órdenes
CREATE TABLE olist_orders (
    order_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64),
    order_status VARCHAR(20),
    order_purchase_timestamp DATETIME,
    order_approved_at DATETIME,
    order_delivered_carrier_date DATETIME,
    order_delivered_customer_date DATETIME,
    order_estimated_delivery_date DATETIME,
    INDEX idx_customer (customer_id),
    INDEX idx_status (order_status),
    INDEX idx_purchase_date (order_purchase_timestamp),
    FOREIGN KEY (customer_id) REFERENCES olist_customers(customer_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de items de órdenes
CREATE TABLE olist_order_items (
    order_id VARCHAR(64),
    order_item_id INT,
    product_id VARCHAR(64),
    seller_id VARCHAR(64),
    shipping_limit_date DATETIME,
    price DECIMAL(10, 2),
    freight_value DECIMAL(10, 2),
    PRIMARY KEY (order_id, order_item_id),
    INDEX idx_product (product_id),
    INDEX idx_seller (seller_id),
    FOREIGN KEY (order_id) REFERENCES olist_orders(order_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (product_id) REFERENCES olist_products(product_id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    FOREIGN KEY (seller_id) REFERENCES olist_sellers(seller_id)
        ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de pagos de órdenes
CREATE TABLE olist_order_payments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id VARCHAR(64),
    payment_sequential INT,
    payment_type VARCHAR(50),
    payment_installments INT,
    payment_value DECIMAL(10, 2),
    INDEX idx_order (order_id),
    INDEX idx_payment_type (payment_type),
    FOREIGN KEY (order_id) REFERENCES olist_orders(order_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de reviews de órdenes
CREATE TABLE olist_order_reviews (
    review_id VARCHAR(64) PRIMARY KEY,
    order_id VARCHAR(64),
    review_score INT,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date DATETIME,
    review_answer_timestamp DATETIME,
    INDEX idx_order (order_id),
    INDEX idx_score (review_score),
    FOREIGN KEY (order_id) REFERENCES olist_orders(order_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
