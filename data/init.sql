CREATE TABLE IF NOT EXISTS prices (
    symbol          VARCHAR(20) NOT NULL,
    date            DATE        NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    adjusted_close  DOUBLE PRECISION,
    volume          BIGINT,
    dividend_amount DOUBLE PRECISION,
    split_coefficient DOUBLE PRECISION,
    PRIMARY KEY (symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_date ON prices (date);
CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices (symbol);
