from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import assets, config
from .model_utils import align_to_booster
from .utils import interval_to_hours, jsonable, parse_dt, uppercase_columns

TARGET_MODE = config.MODEL_A_TARGET_MODE


def _compose_dur(baseline, pred):
    b = np.asarray(baseline, dtype=float)
    p = np.asarray(pred, dtype=float)
    if TARGET_MODE == "log_ratio":
        return np.clip(b, 0.01, None) * np.exp(p)
    return b + np.sign(p) * np.abs(p) ** 3


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


_RIVER_SETS = {
    "GulfCoast": {"Houston", "Beaumont", "Galveston", "Corpus Christi", "Lake Charles",
                  "Port Arthur", "Freeport", "Orange", "Brownsville", "Bayport", "Texas City",
                  "Point Comfort, TX", "Matagorda Bay, TX", "Bay City / Bay Cel", "Harlingen"},
    "IllinoisWaterway": {"Joliet", "Chicago", "Channahon", "Lemont", "Seneca, IL", "Marseilles, IL",
                         "Peru, IL", "PEORIA", "Pekin", "Albany, IL", "Lasalle, IL", "Calumet Sag Channel",
                         "Meredosia, IL"},
    "UpperMiss": {"St. Louis, MO", "St Paul, MN", "Winona, MN", "Savage, MN", "Dubuque, IA",
                  "La Crosse, WI", "Prairie Duchien, WI", "Lousiana, MO", "Burlington",
                  "Clinton", "Keokuk", "Hannibal", "Alton", "Grafton"},
    "Ohio": {"Catlettsburg Harbor", "Mt Vernon, KY", "Louisville", "Cinncinatti", "Pittsburgh, PA",
             "Marietta, OH", "Point Pleasant", "Portsmouth", "Huntington", "Glen Dale, WV",
             "Allegheny River", "LR Evansville, KY", "Owensboro, KY", "Paducah", "Henderson"},
    "TennesseeCumberland": {"Nashville", "Chattanooga", "Decatur, AL", "Florence, AL", "Guntersville",
                            "Tuscaloosa", "Calvert, AL", "New Johnsonville", "Clifton", "Danville, AL"},
    "Arkansas": {"Little Rock", "Catoosa, OK", "Webbers Falls, OK", "Van Buren, AR", "Muskogee",
                 "Fort Smith", "Pine Bluff"},
    "LowerMiss": {"New Orleans, LA", "Baton Rouge", "Cairo, IL", "Memphis", "MERAUX", "Garyville, LA",
                  "GEISMAR", "Darrow, LA", "Sunshine, LA", "ST Gabriel", "Kortz Springs", "Lettsworth",
                  "SL-ORION", "Port of South Louisiana", "Lake Providence, LA", "Greenville, MS",
                  "Helena, AR", "Saint Francisville, LA", "Vicksburg", "Natchez", "Rosedale",
                  "Caruthersville, MO", "New Madrid"},
    "Mobile": {"Mobile", "Pascagoula, MS", "Tampa, FL", "Port Everglades, FL", "New Haven, CT"},
}


def river_system(area_name, lat=None):
    for key, vals in _RIVER_SETS.items():
        if area_name in vals:
            return key
    if lat is not None and not pd.isna(lat):
        if lat >= 37.0:
            return "UpperMiss"
        if lat >= 33.0:
            return "LowerMiss"
        if lat <= 30.5:
            return "GulfCoast"
    return "Other"


class ModelAInfer:
    def __init__(self, client=None, leg_ids=None):
        self.model = lgb.Booster(model_file=str(config.MODEL_FILES["model_a"]))
        self.model_feature_names = list(self.model.feature_name())
        self.route_patterns = assets.read_table(config.MODEL_A_FILES["route_patterns"])
        self.knn_pool = assets.read_table(config.MODEL_A_FILES["knn_pool"])
        self.port_lookup = uppercase_columns(assets.load_port_lookup(client))
        self.vessel_features = uppercase_columns(assets.load_vessel_features(client, leg_ids or []))
        self.subservice_mmsi_map = assets.read_optional(config.MODEL_A_FILES["subservice_mmsi_map"])
        self.carrier_map = assets.read_optional(config.MODEL_A_FILES["carrier_map"])
        self.voyage_extras = assets.read_optional(config.MODEL_A_FILES["voyage_extras"])
        self.context = assets.read_optional(config.FEATURE_FILES["leg_context"])
        self.encoding_priors = assets.read_optional(config.MODEL_A_FILES["encoding_priors"])
        self._prepare_assets()

    def _prepare_assets(self):
        ports = self.port_lookup.copy()
        start_cols = {c: f"START_PORT_{c}" for c in ["NAME", "CITY", "STATE", "COUNTRY", "LON", "LAT"] if c in ports.columns}
        end_cols = {c: f"END_PORT_{c}" for c in ["NAME", "CITY", "STATE", "COUNTRY", "LON", "LAT"] if c in ports.columns}
        if "PORT_AREA" in ports.columns:
            start_cols["PORT_AREA"] = "START_PORT_AREA"
            end_cols["PORT_AREA"] = "END_PORT_AREA"
        self.ports_start = ports.rename(columns={"ID": "START_PORT_ID", **start_cols})
        self.ports_end = ports.rename(columns={"ID": "END_PORT_ID", **end_cols})

        if self.vessel_features is not None:
            self.vessel_features.columns = [str(c).upper() for c in self.vessel_features.columns]

        pool = self.knn_pool.copy()
        pool.columns = [str(c).upper() for c in pool.columns]
        pool["ACTUAL_TRANSIT_H"] = pd.to_numeric(pool.get("ACTUAL_TRANSIT_H"), errors="coerce")
        pool = pool[pool["ACTUAL_TRANSIT_H"].between(1, 1200)].copy()
        pool["MONTH"] = pd.to_datetime(pool.get("DEPARTURE_ANCHOR"), utc=True, errors="coerce").dt.month
        self.knn_pool = pool

        rp = self.route_patterns.copy()
        rp.columns = [str(c).upper() for c in rp.columns]
        self.pattern_global = None
        self.pattern_tables = []
        pat_cols = ["N", "AVG_H", "MED_H", "STD_H", "MED_SPEED", "LOCK_CNT", "LOCK_DELAY"]
        if len(rp):
            for tier in sorted(rp["TIER"].unique()):
                sub = rp[rp["TIER"] == tier]
                if tier == 0:
                    r = sub.iloc[0]
                    self.pattern_global = tuple(float(r[c]) for c in pat_cols)
                    continue
                tbl = {}
                for _, r in sub.iterrows():
                    m = "NA" if pd.isna(r.get("MONTH")) else str(int(r["MONTH"]))
                    key = (("", str(r["START_PORT_AREA"]), "", "", "") if tier == 1 else
                           ("", "", str(r["END_PORT_AREA"]), "", "") if tier == 2 else
                           (str(r["START_PORT_AREA"]), str(r["END_PORT_AREA"]), "", "", "") if tier == 3 else
                           (str(r["START_PORT_AREA"]), str(r["END_PORT_AREA"]), m, "", "") if tier == 4 else
                           (str(r["START_PORT_AREA"]), str(r["END_PORT_AREA"]), m, str(r["SVC"]), "") if tier == 5 else
                           (str(r["START_PORT_AREA"]), str(r["END_PORT_AREA"]), m, str(r["SVC"]), str(r["SHIP"])))
                    tbl[key] = tuple(float(r[c]) if pd.notna(r[c]) else np.nan for c in pat_cols)
                self.pattern_tables.append(tbl)

        if self.encoding_priors is not None:
            pri = self.encoding_priors.copy()
            pri.columns = [str(c).lower() for c in pri.columns]
            self.pp = pri[pri.get("level").eq("port_pair")].copy() if "level" in pri.columns else None
        else:
            self.pp = None

        if self.carrier_map is not None:
            self.carrier_map.columns = [str(c).upper() for c in self.carrier_map.columns]
        if self.voyage_extras is not None:
            self.voyage_extras.columns = [str(c).upper() for c in self.voyage_extras.columns]
        if self.context is not None:
            self.context.columns = [str(c).lower() for c in self.context.columns]

    def _enrich(self, legs_df: pd.DataFrame) -> pd.DataFrame:
        d = uppercase_columns(legs_df)
        for c in ["TIME_STARTED", "TIME_COMPLETED", "INITIAL_ETA", "TIME_CREATED", "ETA_UPDATED", "EARLIEST_DEPARTURE", "LATEST_DEPARTURE", "EARLIEST_ARRIVAL", "LATEST_ARRIVAL"]:
            if c in d.columns:
                d[c] = parse_dt(d[c])
        d = d.merge(self.ports_start, on="START_PORT_ID", how="left")
        d = d.merge(self.ports_end, on="END_PORT_ID", how="left")
        if "SUBSERVICE_ID" in self.vessel_features.columns:
            vess = self.vessel_features.rename(columns={"SUBSERVICE_ID": "ID"})
            d = d.merge(vess, on="ID", how="left", suffixes=("", "_VESSEL"))
        if self.carrier_map is not None and "VOYAGE_ID" in d.columns and "VOYAGE_ID" in self.carrier_map.columns:
            d = d.merge(self.carrier_map.drop_duplicates("VOYAGE_ID"), on="VOYAGE_ID", how="left")
        if self.voyage_extras is not None and "VOYAGE_ID" in d.columns and "VOYAGE_ID" in self.voyage_extras.columns:
            d = d.merge(self.voyage_extras.drop_duplicates("VOYAGE_ID"), on="VOYAGE_ID", how="left", suffixes=("", "_EXTRA"))
        if self.context is not None and "ID" in d.columns:
            d = d.merge(self.context.rename(columns={"subservice_id": "ID"}), on="ID", how="left")
            if "has_context" in d.columns:
                d["has_context"] = d["has_context"].fillna(0).astype(float)
        return d

    @staticmethod
    def parse_route_all(val):
        keys = ("route_dur_h", "route_dist_nm", "route_locks_n", "route_delay_sum_h", "route_process_sum_h", "route_min_width", "route_min_length", "route_rivers_n", "route_giww")
        out = dict.fromkeys(keys, np.nan)
        if pd.isna(val):
            return out
        try:
            r = json.loads(val)
        except Exception:
            return out
        if not isinstance(r, dict):
            return out
        out["route_dur_h"] = interval_to_hours(r.get("duration"))
        out["route_dist_nm"] = pd.to_numeric(r.get("distance"), errors="coerce")
        locks = r.get("locks") or []
        out["route_locks_n"] = float(len(locks))
        delays, procs, ws, ls, rivers = [], [], [], [], set()
        giww = 0.0
        for l in locks:
            if not isinstance(l, dict):
                continue
            d1 = interval_to_hours(l.get("average_delay"))
            if not pd.isna(d1):
                delays.append(d1)
            p1 = interval_to_hours(l.get("avg_process"))
            if not pd.isna(p1):
                procs.append(p1)
            try:
                ws.append(float(l.get("width")))
            except Exception:
                pass
            try:
                ls.append(float(l.get("length")))
            except Exception:
                pass
            rv = l.get("river") or {}
            code = rv.get("code")
            if code:
                rivers.add(code)
                if str(code).upper() == "GI":
                    giww = 1.0
        if delays:
            out["route_delay_sum_h"] = float(np.nansum(delays))
        if procs:
            out["route_process_sum_h"] = float(np.nansum(procs))
        if ws:
            out["route_min_width"] = float(np.nanmin(ws))
        if ls:
            out["route_min_length"] = float(np.nanmin(ls))
        out["route_rivers_n"] = float(len(rivers))
        out["route_giww"] = giww
        return out

    def _apply_pattern_features(self, d: pd.DataFrame) -> pd.DataFrame:
        cols = ["pat_avg_h", "pat_med_h", "pat_std_h", "pat_count", "pat_speed_nm_h", "pat_lock_cnt", "pat_lock_delay_h"]
        d[cols] = np.nan
        if not self.pattern_tables and self.pattern_global is None:
            return d
        out = []
        for _, r in d.iterrows():
            month = "NA" if pd.isna(r.get("dep_month")) else str(int(r.get("dep_month")))
            ship = str(r.get("ship_class", "")) if pd.notna(r.get("ship_class")) else ""
            svc = str(r.get("SERVICE_TYPE", ""))
            keys = [
                (str(r.get("START_PORT_AREA")), str(r.get("END_PORT_AREA")), month, svc, ship),
                (str(r.get("START_PORT_AREA")), str(r.get("END_PORT_AREA")), month, svc, ""),
                (str(r.get("START_PORT_AREA")), str(r.get("END_PORT_AREA")), month, "", ""),
                (str(r.get("START_PORT_AREA")), str(r.get("END_PORT_AREA")), "", "", ""),
                ("", "", str(r.get("END_PORT_AREA")), "", ""),
                ("", str(r.get("START_PORT_AREA")), "", "", ""),
            ]
            hit = None
            for key in keys:
                for tbl in reversed(self.pattern_tables):
                    if key in tbl:
                        hit = tbl[key]
                        break
                if hit is not None:
                    break
            if hit is None:
                hit = self.pattern_global or (np.nan,) * 7
            out.append(hit)
        pats = pd.DataFrame(out, columns=cols, index=d.index)
        for c in cols:
            d[c] = pats[c]
        d["pat_ratio_naive"] = d["pat_med_h"] / (d["naive_h"] + 0.001)
        return d

    def _apply_knn_features(self, d: pd.DataFrame) -> pd.DataFrame:
        pool = self.knn_pool
        out = []
        for _, r in d.iterrows():
            sub = pool[(pool["START_PORT_AREA"].astype(str) == str(r.get("START_PORT_AREA")))
                       & (pool["END_PORT_AREA"].astype(str) == str(r.get("END_PORT_AREA")))]
            if "SERVICE_TYPE" in pool.columns and pd.notna(r.get("SERVICE_TYPE")):
                sub2 = sub[sub["SERVICE_TYPE"].astype(str) == str(r.get("SERVICE_TYPE"))]
                if len(sub2):
                    sub = sub2
            if len(sub) == 0:
                out.append((np.nan, np.nan, np.nan, np.nan, 0.0))
                continue
            dist = pd.to_numeric(sub.get("ESTIMATED_DISTANCE"), errors="coerce")
            speed = pd.to_numeric(sub.get("ESTIMATED_SPEED"), errors="coerce")
            dvec = np.sqrt((dist - pd.to_numeric(r.get("ESTIMATED_DISTANCE"), errors="coerce")) ** 2 + (speed - pd.to_numeric(r.get("ESTIMATED_SPEED"), errors="coerce")) ** 2)
            sub = sub.assign(_d=dvec).sort_values("_d").head(30)
            h = pd.to_numeric(sub["ACTUAL_TRANSIT_H"], errors="coerce").dropna()
            out.append((float(h.mean()) if len(h) else np.nan,
                        float(h.median()) if len(h) else np.nan,
                        float(h.std()) if len(h) else np.nan,
                        float(sub["_d"].mean()) if len(sub) else np.nan,
                        float(len(sub))))
        k = pd.DataFrame(out, columns=["knn_pred_h", "knn_med_h", "knn_std_h", "knn_mean_d", "knn_n_tight"], index=d.index)
        for c in k.columns:
            d[c] = k[c]
        d["knn_ratio_naive"] = d["knn_pred_h"] / (d["naive_h"] + 0.001)
        return d

    def _apply_encoding_priors(self, d: pd.DataFrame) -> pd.DataFrame:
        if self.pp is None or len(self.pp) == 0:
            gm, gs = 0.0, 1.0
            d["pair_hist_resid"] = gm
            d["pair_hist_std"] = gs
            d["pair_hist_count"] = 1.0
            d["pair_hist_iqr"] = gs
            d["pair_hist_median"] = gm
            d["pair_hist_cv"] = gs / (abs(gm) + 0.001)
            d["start_port_hist"] = gm
            d["end_port_hist"] = gm
            d["carrier_hist_resid"] = gm
            d["carrier_hist_std"] = gs
            d["carrier_hist_count"] = 0.0
            d["pair_hist_resid_by_svc"] = gm
            d["start_port_hist_by_svc"] = gm
            d["end_port_hist_by_svc"] = gm
            d["pair_svc_count"] = 0.0
        else:
            shortcut = self.pp.set_index("key")
            gm = float(shortcut.get("global_mean", pd.Series([0.0])).iloc[0]) if "global_mean" in shortcut.columns else 0.0
            gs = float(shortcut.get("global_std", pd.Series([1.0])).iloc[0]) if "global_std" in shortcut.columns else 1.0
            pp = shortcut[shortcut["kind"].eq("pair")] if "kind" in shortcut.columns else shortcut
            d["pair_hist_resid"] = d["port_pair"].map(pp.get("pair_mean")).fillna(gm)
            d["pair_hist_std"] = d["port_pair"].map(pp.get("pair_std")).fillna(gs)
            d["pair_hist_count"] = d["port_pair"].map(pp.get("pair_count")).fillna(1.0)
            d["pair_hist_iqr"] = d["port_pair"].map(pp.get("pair_iqr")).fillna(gs)
            d["pair_hist_median"] = d["port_pair"].map(pp.get("pair_median")).fillna(gm)
            d["pair_hist_cv"] = d["pair_hist_std"] / (d["pair_hist_resid"].abs() + 0.001)
            d["start_port_hist"] = gm
            d["end_port_hist"] = gm
            d["carrier_hist_resid"] = gm
            d["carrier_hist_std"] = gs
            d["carrier_hist_count"] = 0.0
            d["pair_hist_resid_by_svc"] = gm
            d["start_port_hist_by_svc"] = gm
            d["end_port_hist_by_svc"] = gm
            d["pair_svc_count"] = 0.0
        d["pair_hist_resid_x_count"] = d["pair_hist_resid"] * d["pair_hist_count"]
        d["pair_hist_median_x_count"] = d["pair_hist_median"] * d["pair_hist_count"]
        return d

    @staticmethod
    def _season4(months: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [months.isin([12, 1, 2]), months.isin([3, 4, 5]), months.isin([6, 7, 8])],
                ["winter", "spring", "summer"],
                default="fall",
            ),
            index=months.index,
        )

    def _prepare_model_frame(self, feat: pd.DataFrame) -> pd.DataFrame:
        model_frame = feat.copy()

        if "RIVER_SYSTEM" in model_frame.columns and model_frame["RIVER_SYSTEM"].isna().all():
            model_frame["RIVER_SYSTEM"] = [
                river_system(a, la) for a, la in zip(feat.get("START_PORT_AREA"), feat.get("START_PORT_LAT"))
            ]
        if "season4" in model_frame.columns and model_frame["season4"].isna().all() and "dep_month" in feat.columns:
            model_frame["season4"] = self._season4(feat["dep_month"])
        if "season_river" in model_frame.columns and model_frame["season_river"].isna().all():
            season4 = model_frame.get("season4")
            river = model_frame.get("RIVER_SYSTEM")
            if season4 is not None and river is not None:
                model_frame["season_river"] = season4.astype(str) + "_" + river.astype(str)
        return align_to_booster(model_frame, self.model, feature_names=self.model_feature_names)

    def build_features(self, legs_df: pd.DataFrame) -> pd.DataFrame:
        d = self._enrich(legs_df)
        if "SELECTED_ROUTE" in d.columns:
            ra = d["SELECTED_ROUTE"].apply(self.parse_route_all).apply(pd.Series)
            for c in ra.columns:
                d[c] = ra[c]
        d["LOCKS_HOURS_INCLUDED"] = pd.to_numeric(d.get("LOCKS_HOURS_INCLUDED"), errors="coerce").fillna(0.0)
        d["dwell_start_h"] = d.get("TIME_IN_START_PORT", pd.Series(index=d.index)).apply(interval_to_hours).fillna(12.0)
        d["dwell_end_h"] = d.get("TIME_IN_END_PORT", pd.Series(index=d.index)).apply(interval_to_hours).fillna(12.0)
        d["naive_h"] = pd.to_numeric(d.get("ESTIMATED_DISTANCE"), errors="coerce") / pd.to_numeric(d.get("ESTIMATED_SPEED"), errors="coerce").replace(0, np.nan) + d["LOCKS_HOURS_INCLUDED"]
        d["baseline_h"] = d["naive_h"]
        d["has_route_dur"] = d.get("route_dur_h", pd.Series(index=d.index)).notna().astype(int)
        d["plan_anchor"] = d.get("TIME_CREATED")
        d["dep_month"] = d["plan_anchor"].dt.month
        d["dep_dow"] = d["plan_anchor"].dt.dayofweek
        d["dep_hour"] = d["plan_anchor"].dt.hour
        d["dist_per_lock"] = pd.to_numeric(d.get("ESTIMATED_DISTANCE"), errors="coerce") / (pd.to_numeric(d.get("NUM_LOCKS"), errors="coerce").fillna(0) + 1)
        d["speed_x_dist"] = pd.to_numeric(d.get("ESTIMATED_SPEED"), errors="coerce") * pd.to_numeric(d.get("ESTIMATED_DISTANCE"), errors="coerce")
        d["dwell_total_h"] = d["dwell_start_h"] + d["dwell_end_h"]
        d["route_vs_naive"] = pd.to_numeric(d.get("route_dist_nm"), errors="coerce") / (pd.to_numeric(d.get("ESTIMATED_DISTANCE"), errors="coerce") + 0.001)
        d["port_pair"] = d["START_PORT_ID"].astype(str) + "_" + d["END_PORT_ID"].astype(str)
        d["idle_ratio"] = pd.to_numeric(d.get("IDLE_FUEL_PER_DAY"), errors="coerce") / (pd.to_numeric(d.get("FUEL_BURN_PER_DAY"), errors="coerce") + 0.001)
        d["created_to_start_h"] = (d["ETA_UPDATED"] - d["TIME_CREATED"]).dt.total_seconds().div(3600) if "ETA_UPDATED" in d.columns else np.nan
        d["dist_per_lock_sq"] = pd.to_numeric(d.get("ESTIMATED_DISTANCE"), errors="coerce") / (pd.to_numeric(d.get("NUM_LOCKS"), errors="coerce").fillna(0) ** 2 + 1)
        d["svc_x_speed"] = pd.to_numeric(d.get("SERVICE_TYPE"), errors="coerce") * pd.to_numeric(d.get("ESTIMATED_SPEED"), errors="coerce")
        d["dep_hour_x_dist"] = d["dep_hour"] * pd.to_numeric(d.get("ESTIMATED_DISTANCE"), errors="coerce")
        d["dep_month_x_dist"] = d["dep_month"] * pd.to_numeric(d.get("ESTIMATED_DISTANCE"), errors="coerce")
        d["bearing_deg"] = _bearing_deg(pd.to_numeric(d.get("START_PORT_LAT"), errors="coerce"), pd.to_numeric(d.get("START_PORT_LON"), errors="coerce"), pd.to_numeric(d.get("END_PORT_LAT"), errors="coerce"), pd.to_numeric(d.get("END_PORT_LON"), errors="coerce"))
        d["lat_midpoint"] = (pd.to_numeric(d.get("START_PORT_LAT"), errors="coerce") + pd.to_numeric(d.get("END_PORT_LAT"), errors="coerce")) / 2
        d["route_dur_vs_naive"] = pd.to_numeric(d.get("route_dur_h"), errors="coerce") - d["naive_h"]
        d["edited_route_flag"] = d.get("EDITED_ROUTE", False).fillna(False).astype(int)
        d["tow_size_x_distance"] = pd.to_numeric(d.get("TOW_SIZE"), errors="coerce") * pd.to_numeric(d.get("ESTIMATED_DISTANCE"), errors="coerce")
        d["deadweight_x_speed"] = pd.to_numeric(d.get("TOTAL_DEADWEIGHT"), errors="coerce") * pd.to_numeric(d.get("ESTIMATED_SPEED"), errors="coerce")
        d["dep_month_x_route_dur"] = d["dep_month"] * pd.to_numeric(d.get("route_dur_h"), errors="coerce").fillna(d["naive_h"])
        d["bearing_x_speed"] = d["bearing_deg"] * pd.to_numeric(d.get("ESTIMATED_SPEED"), errors="coerce")
        if "EARLIEST_DEPARTURE" in d.columns and "LATEST_DEPARTURE" in d.columns:
            d["departure_window_h"] = (d["LATEST_DEPARTURE"] - d["EARLIEST_DEPARTURE"]).dt.total_seconds() / 3600
        if "EARLIEST_ARRIVAL" in d.columns and "LATEST_ARRIVAL" in d.columns:
            d["arrival_window_h"] = (d["LATEST_ARRIVAL"] - d["EARLIEST_ARRIVAL"]).dt.total_seconds() / 3600
        m = d["dep_month"]
        d["month_sin"] = np.sin(2 * np.pi * m / 12.0)
        d["month_cos"] = np.cos(2 * np.pi * m / 12.0)
        d["upstream_flag"] = (((d["bearing_deg"] >= 315) | (d["bearing_deg"] < 45))).astype(int)
        d["low_water"] = 0
        d["spring_flood"] = 0
        d["hurricane_gulf"] = 0
        d["mon_x_locks"] = m * pd.to_numeric(d.get("NUM_LOCKS"), errors="coerce")
        svc01 = pd.to_numeric(d["SERVICE_TYPE"], errors="coerce").eq(2).astype(float) if "SERVICE_TYPE" in d.columns else 0.0
        d["svc_x_naive"] = svc01 * d["naive_h"]
        d["svc_x_locks"] = svc01 * pd.to_numeric(d.get("NUM_LOCKS"), errors="coerce")
        d["svc_x_tow"] = svc01 * pd.to_numeric(d.get("TOW_SIZE"), errors="coerce").fillna(0)
        d["svc_x_dwell"] = svc01 * d["dwell_total_h"]
        d["start_standby_h"] = d.get("START_STANDBY_TIME", pd.Series(index=d.index)).apply(interval_to_hours)
        d["end_standby_h"] = d.get("END_STANDBY_TIME", pd.Series(index=d.index)).apply(interval_to_hours)
        d["barge_slots"] = pd.to_numeric(d.get("BARGE_SLOTS"), errors="coerce")
        d["util_weight"] = np.where(pd.to_numeric(d.get("CAPACITY_WEIGHT"), errors="coerce") > 0, pd.to_numeric(d.get("TOTAL_LOADED_TONS"), errors="coerce") / pd.to_numeric(d.get("CAPACITY_WEIGHT"), errors="coerce"), np.nan)
        d["util_vol"] = np.where(pd.to_numeric(d.get("CAPACITY_BARRELS"), errors="coerce") > 0, pd.to_numeric(d.get("TOTAL_LOADED_BARRELS"), errors="coerce") / pd.to_numeric(d.get("CAPACITY_BARRELS"), errors="coerce"), np.nan)
        d["local_dep_hour"] = d["dep_hour"]
        d["is_night_local"] = d["local_dep_hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        d["port_pop_log"] = np.nan
        d = self._apply_pattern_features(d)
        d = self._apply_knn_features(d)
        d = self._apply_encoding_priors(d)
        return d

    def predict(self, legs_df: pd.DataFrame):
        feat = self.build_features(legs_df)
        model_frame = self._prepare_model_frame(feat)
        pred = self.model.predict(model_frame)
        feat["predicted_duration_h"] = _compose_dur(feat["baseline_h"], pred)
        out_cols = [c for c in ["ID", "VOYAGE_ID", "SEQUENCE_NUM", "predicted_duration_h"] if c in feat.columns]
        out = feat[out_cols].copy()
        return out, {int(r["ID"]): jsonable(r.to_dict()) for _, r in feat.iterrows() if pd.notna(r.get("ID"))}
