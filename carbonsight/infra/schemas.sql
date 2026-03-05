-- CarbonSight MVP schema per design doc
-- cloud_region, cloud_site, grid_mapping, run, grid_signal_cache

CREATE TABLE IF NOT EXISTS cloud_region (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(32) NOT NULL,
    region_code VARCHAR(32) NOT NULL,
    display_name VARCHAR(255),
    country VARCHAR(64),
    source_url TEXT,
    UNIQUE(provider, region_code)
);

CREATE TABLE IF NOT EXISTS cloud_site (
    site_id VARCHAR(64) PRIMARY KEY,
    provider VARCHAR(32) NOT NULL,
    region_code VARCHAR(32) NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    source_type VARCHAR(32) DEFAULT 'official',
    last_verified_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS grid_mapping (
    site_id VARCHAR(64) NOT NULL REFERENCES cloud_site(site_id),
    signal_type VARCHAR(32) NOT NULL DEFAULT 'co2_moer',
    wt_region VARCHAR(32) NOT NULL,
    validated_at TIMESTAMPTZ,
    confidence_subscores JSONB,
    PRIMARY KEY (site_id, signal_type)
);

CREATE TABLE IF NOT EXISTS run (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    job_spec_json JSONB NOT NULL,
    chosen_region VARCHAR(64),
    estimated_co2_kg DOUBLE PRECISION,
    estimated_cost_usd DOUBLE PRECISION,
    sky_job_id VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS grid_signal_cache (
    wt_region VARCHAR(32) NOT NULL,
    signal_type VARCHAR(32) NOT NULL,
    point_time TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    units VARCHAR(32),
    data_point_period_seconds INT,
    model_date DATE,
    PRIMARY KEY (wt_region, signal_type, point_time)
);

CREATE INDEX IF NOT EXISTS idx_run_submitted ON run(submitted_at);
CREATE INDEX IF NOT EXISTS idx_grid_signal_region_time ON grid_signal_cache(wt_region, point_time);
