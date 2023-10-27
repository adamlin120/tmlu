def topic_line_template(content):
    return f"以下題目為選擇題，{content}，答案可能為單個或多個。"

topic_line_mapping = {
    "GSAT_chinese.jsonl": topic_line_template("出自臺灣學科能力測驗國文科"),
    "GSAT_civics.jsonl": topic_line_template("出自臺灣學科能力測驗社會科"),
    "GSAT_history.jsonl": topic_line_template("出自臺灣學科能力測驗社會科"),
    "GSAT_geography.jsonl": topic_line_template("出自臺灣學科能力測驗社會科"),
    "AST_civic.jsonl": topic_line_template("出自臺灣分科測驗公民與社會科"),
    "AST_history.jsonl": topic_line_template("出自臺灣分科測驗歷史科"),
    "AST_geography.jsonl": topic_line_template("出自臺灣分科測驗地理科"),
    "AST_biology.jsonl": topic_line_template("出自臺灣分科測驗生物科"),
    "AST_chemistry.jsonl": topic_line_template("出自臺灣分科測驗化學科"),
    "AST_physics.jsonl": topic_line_template("出自臺灣分科測驗物理科")
}