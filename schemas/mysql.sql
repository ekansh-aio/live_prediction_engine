CREATE TABLE IF NOT EXISTS voyage_prediction_runs (
  run_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  voyage_id BIGINT NOT NULL,
  as_of_utc DATETIME NOT NULL,
  current_leg_id BIGINT NOT NULL,
  consolidated_eta_utc DATETIME NULL,
  eta_p10_utc DATETIME NULL,
  eta_p90_utc DATETIME NULL,
  total_remaining_h DOUBLE NULL,
  raw_total_remaining_h DOUBLE NULL,
  bias_adj_h DOUBLE NULL,
  current_remaining_h DOUBLE NULL,
  future_legs_duration_h DOUBLE NULL,
  future_dwell_h DOUBLE NULL,
  current_marker_wait_h DOUBLE NULL,
  future_marker_wait_h DOUBLE NULL,
  result_json JSON NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS voyage_prediction_legs (
  row_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  run_id BIGINT NOT NULL,
  voyage_id BIGINT NOT NULL,
  leg_id BIGINT NOT NULL,
  sequence_num INT NOT NULL,
  role VARCHAR(32) NOT NULL,
  leg_status VARCHAR(32) NOT NULL,
  model_a_pred_h DOUBLE NULL,
  dynamic_pred_h DOUBLE NULL,
  flc_residual_h DOUBLE NULL,
  flc_corrected_h DOUBLE NULL,
  model_a_features_json JSON NULL,
  dynamic_features_json JSON NULL,
  flc_features_json JSON NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_run_id (run_id)
);
