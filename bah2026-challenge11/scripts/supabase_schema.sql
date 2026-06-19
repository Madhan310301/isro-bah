-- Run this in the Supabase SQL editor to set up the satellite_images table.

CREATE TABLE IF NOT EXISTS satellite_images (
  id            BIGSERIAL PRIMARY KEY,
  pair_id       TEXT        NOT NULL,
  modality      TEXT        CHECK (modality IN ('sar', 'optical')),
  season        TEXT,
  storage_path  TEXT,
  lat           FLOAT8,
  lon           FLOAT8,
  embedding_id  BIGINT,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Indices for fast lookups by pair_id and FAISS embedding_id
CREATE INDEX IF NOT EXISTS idx_satellite_images_pair_id       ON satellite_images (pair_id);
CREATE INDEX IF NOT EXISTS idx_satellite_images_embedding_id  ON satellite_images (embedding_id);
CREATE INDEX IF NOT EXISTS idx_satellite_images_modality      ON satellite_images (modality);

-- Storage bucket (run via Supabase dashboard or management API)
-- INSERT INTO storage.buckets (id, name, public)
-- VALUES ('satellite-images', 'satellite-images', true)
-- ON CONFLICT DO NOTHING;
