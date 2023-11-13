def topic_line_template(content):
    return f"以下題目為選擇題，{content}，答案可能為單個或多個。"

topic_line_mapping = {
    "GSAT_chinese_test.jsonl": topic_line_template("出自臺灣學科能力測驗國文科"),
    "GSAT_civics_test.jsonl": topic_line_template("出自臺灣學科能力測驗社會科"),
    "GSAT_history_test.jsonl": topic_line_template("出自臺灣學科能力測驗社會科"),
    "GSAT_geography_test.jsonl": topic_line_template("出自臺灣學科能力測驗社會科"),
    "GSAT_biology_test.jsonl": topic_line_template("出自臺灣學科能力測驗自然科"),
    "GSAT_chemistry_test.jsonl": topic_line_template("出自臺灣學科能力測驗自然科"),
    "GSAT_physics_test.jsonl": topic_line_template("出自臺灣學科能力測驗自然科"),
    "GSAT_earth_science_test.jsonl": topic_line_template("出自臺灣學科能力測驗自然科"),
    "AST_chinese_test.jsonl": topic_line_template("出自臺灣分科測驗國文科"),
    "AST_civics_test.jsonl": topic_line_template("出自臺灣分科測驗公民與社會科"),
    "AST_history_test.jsonl": topic_line_template("出自臺灣分科測驗歷史科"),
    "AST_geography_test.jsonl": topic_line_template("出自臺灣分科測驗地理科"),
    "AST_biology_test.jsonl": topic_line_template("出自臺灣分科測驗生物科"),
    "AST_chemistry_test.jsonl": topic_line_template("出自臺灣分科測驗化學科"),
    "AST_physics_test.jsonl": topic_line_template("出自臺灣分科測驗物理科"),
    "CAP_chinese_test.jsonl": topic_line_template("出自臺灣國中教育會考國文科"),
    "CAP_mathematics_test.jsonl": topic_line_template("出自臺灣國中教育會考數學科"),
    "CAP_biology_test.jsonl": topic_line_template("出自臺灣國中教育會考自然科"),
    "CAP_chemistry_test.jsonl": topic_line_template("出自臺灣國中教育會考自然科"),
    "CAP_physics_test.jsonl": topic_line_template("出自臺灣國中教育會考自然科"),
    "CAP_earth_science_test.jsonl": topic_line_template("出自臺灣國中教育會考自然科"),
    "CAP_geography_test.jsonl": topic_line_template("出自臺灣國中教育會考社會科"),
    "CAP_history_test.jsonl": topic_line_template("出自臺灣國中教育會考社會科"),
    "CAP_civics_test.jsonl": topic_line_template("出自臺灣國中教育會考社會科"),
}