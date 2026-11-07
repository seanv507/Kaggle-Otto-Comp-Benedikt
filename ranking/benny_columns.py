"""Benny's column lists from 02_Clicks/01_Split.ipynb, copied as they are.

His rankers drop a feature column if it's in all three of his per-target ignore
lists, plus some chris__<i>__ columns. The lists include names that no source writes
(e.g. weighted_0_2_*, baseline); they're kept as he had them.
"""

CLICKS_IGNORE = [
    "weighted_0_2_dummy", "weighted_0_2_count", "num_aid_ca",
    "num_aid_ca_fs", "c2o_num_15", "weighted_0_2_wgt_sum_2",
    "num_aid_fs", "b2b_rank_min_15", "caco_wgt_ca_15", "b2b_wgt_ca_15",
    "clicks_wgt_or_20", "num_u_aid_cl_fs", "caco_wgt_15",
    "b2b_wgt_2_15", "carts", "caco_wgt_2_15", "c2o_wgt_ca_15",
    "clicks_wgt_ca_20", "orders", "c2o_wgt_or_15",
    "weighted_0_2_wgt_prob_2", "num_sessions", "caco_b2b_rank_min_15",
    "caco_b2b_wgt_15", "caco_b2b_wgt_2_15", "caco_b2b_wgt_cl_15",
    "caco_b2b_wgt_ca_15", "caco_b2b_wgt_or_15", "num_aid_co",
    "caco_b2b_num_15", "num_u_aid_ca_fs", "num_aid_co_fs",
    "num_u_aid_co", "num_u_aid_co_fs", "b2b_wgt_or_15", "b2b_wgt_15",
    "weighted_0_2_wgt_sum_2_2", "caco_wgt_or_15", "b2b_num_15",
    "c2o_wgt_2_15", "caco_num_15", "clicks_num_20", "b2b_wgt_cl_15",
    "weighted_0_2_count_2", "c2o_wgt_cl_15", "weighted_0_2_wgt_prob",
    "num_u_aid_ca", "num_aid_cl_fs", "caco_wgt_cl_15",
]  # fmt: skip

CARTS_IGNORE = [
    "num_sessions", "caco_wgt_15", "b2b_rank_min_15", "b2b_wgt_ca_15",
    "clicks_wgt_2_20", "num_aid_co", "weighted_0_2_dummy",
    "caco_num_15", "ts_diff", "num_aid_fs", "caco_wgt_cl_15",
    "c2o_wgt_cl_15", "num_u_aid_co_fs", "num_aid_co_fs",
    "caco_wgt_or_15", "c2o_wgt_or_15", "caco_wgt_ca_15",
    "num_u_aid_co", "baseline", "caco_b2b_wgt_ca_15",
    "caco_b2b_wgt_cl_15", "caco_b2b_wgt_2_15", "caco_b2b_wgt_15",
    "caco_b2b_rank_min_15", "caco_b2b_num_15", "orders",
    "caco_b2b_wgt_or_15", "baseline2", "b2b_wgt_or_15",
    "clicks_wgt_or_20", "c2o_wgt_ca_15", "clicks_wgt_ca_20",
    "weighted_0_2_count", "num_u_aid_cl_fs",
]  # fmt: skip

ORDERS_IGNORE = [
    "clicks_num_20", "num_u_aid", "num_aid_ca", "caco_b2b_wgt_or_15",
    "baseline", "caco_b2b_wgt_cl_15", "caco_rank_min_15",
    "caco_wgt_or_15", "caco_b2b_wgt_ca_15", "orders", "baseline2",
    "caco_b2b_rank_min_15", "caco_b2b_wgt_15", "caco_b2b_wgt_2_15",
    "caco_b2b_num_15", "b2b_wgt_ca_15", "num_sessions",
    "caco_wgt_ca_15", "num_aid_cl_fs", "weighted_count_30",
    "num_u_aid_co_fs", "caco_wgt_15", "weighted_0_2_count",
    "num_aid_co_fs",
]  # fmt: skip

# total_ignore: in all three lists (in ORDERS_IGNORE's order), then the chris columns
# `mainprefix + '_' + str(i) + '_' + x` with mainprefix 'chris_'
TOTAL_IGNORE = [
    x for x in ORDERS_IGNORE if x in CARTS_IGNORE and x in CLICKS_IGNORE
] + [
    f"chris__{i}__{x.lstrip('_')}"
    for i in range(15)
    for x in ["_l_dummy", "_l_wgt_2", "_l_dummy", "_d_wgt_1", "_t_wgt_2", "_t_dummy"]
]

# the sources whose candidates the clicks ranker takes (get_cands), each ranked by
# the one column of its file that is in Benny's list_sort_columns
CLICKS_GENERATORS = {
    "c2o_15": "c2o_wgt_15",
    "clicks_20": "clicks_wgt_20",
    "b2b_15": "b2b_wgt_15",
    "caco_15": "caco_wgt_15",
    "pv": "pv_wgt_1",
    "weighted_30": "weighted_wgt_30",
    "caco_b2b_15": "caco_b2b_wgt_15",
}
NO_CANDS = 80  # per generating source

# Benny's list_sort_columns: a source generates candidates ranked by the one of its
# columns in this list (the carts and orders rankers take every source but the GRU's)
LIST_SORT_COLUMNS = [
    "c2o_wgt_15", "clicks_wgt_20", "b2b_wgt_15", "caco_wgt_15", "pv_wgt_1",
    "weighted_wgt_30", "weighted_1_2_wgt_sum_2", "t4r_score", "caco_b2b_wgt_15",
    "pv_2_wgt_1", "caco_df__wgt_15",
    *[f"chris__{i}__wgt_1" for i in range(10)],
]  # fmt: skip
